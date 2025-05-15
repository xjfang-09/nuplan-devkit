from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Generator, List, Optional, Set, Tuple, Union, cast

import nuplan.database.nuplan_db.image as ImageDBRow
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.geometry.interpolate_state import interpolate_future_waypoints
from nuplan.database.common.blob_store.creator import BlobStoreCreator
from nuplan.database.common.blob_store.local_store import LocalStore
from nuplan.database.common.blob_store.s3_store import S3Store
from nuplan.database.nuplan_db.lidar_pc import LidarPc
from nuplan.database.nuplan_db.nuplan_db_utils import SensorDataSource, get_lidarpc_sensor_data
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_future_waypoints_for_agents_from_db,
    get_sampled_sensor_tokens_in_time_window_from_db,
    get_sensor_data_token_timestamp_from_db,
    get_tracked_objects_for_lidarpc_token_from_db,
    get_tracked_objects_within_time_interval_from_db,
)
from nuplan.database.utils.image import Image
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

logger = logging.getLogger(__name__)

LIDAR_PC_CACHE = 16 * 2**10  # 16K 缓存大小

DEFAULT_SCENARIO_NAME = 'unknown'  # 场景名称（如：自车超车）
DEFAULT_SCENARIO_DURATION = 20.0  # [s] 场景持续时间（如：事件发生后提取20秒数据）
DEFAULT_EXTRACTION_OFFSET = 0.0  # [s] 场景起始偏移量（如：从事件前5秒开始）
DEFAULT_SUBSAMPLE_RATIO = 1.0  # 子采样比率（如：0.1 表示从20Hz降采样到2Hz）

NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', "/data/sets/nuplan/")


@dataclass(frozen=True)
class ScenarioExtractionInfo:
    """
    包含用于提取场景（lidarpc 序列）的信息。
    """

    scenario_name: str = DEFAULT_SCENARIO_NAME  # 场景名称
    scenario_duration: float = DEFAULT_SCENARIO_DURATION  # [s] 场景持续时间
    extraction_offset: float = DEFAULT_EXTRACTION_OFFSET  # [s] 场景起始偏移量
    subsample_ratio: float = DEFAULT_SUBSAMPLE_RATIO  # 场景采样比率

    def __post_init__(self) -> None:
        """校验类属性"""
        assert 0.0 < self.scenario_duration, f"场景持续时间必须大于0，当前值：{self.scenario_duration}"
        assert 0.0 < self.subsample_ratio <= 1.0, f"子采样率需在0~1之间，当前值：{self.subsample_ratio}"


class ScenarioMapping:
    """
    将每种场景类型映射到对应的提取指令。
    """

    def __init__(
        self,
        scenario_map: Dict[str, Union[Tuple[float, float, float], Tuple[float, float]]],
        subsample_ratio_override: Optional[float],
    ) -> None:
        """
        初始化场景映射类。
        :param scenario_map: 字典，键是场景名称/类型，值是 (持续时间, 偏移量, 子采样率) 的元组。
        :param subsample_ratio_override: 如果提供了此参数，则会覆盖默认的子采样率。
        """
        self.mapping: Dict[str, ScenarioExtractionInfo] = {}
        self.subsample_ratio_override = (
            subsample_ratio_override if subsample_ratio_override is not None else DEFAULT_SUBSAMPLE_RATIO
        )

        for name in scenario_map:
            this_ratio: float = scenario_map[name][2] if len(scenario_map[name]) == 3 else self.subsample_ratio_override  # type: ignore

            self.mapping[name] = ScenarioExtractionInfo(
                scenario_name=name,
                scenario_duration=scenario_map[name][0],
                extraction_offset=scenario_map[name][1],
                subsample_ratio=this_ratio,
            )

    def get_extraction_info(self, scenario_type: str) -> Optional[ScenarioExtractionInfo]:
        """
        根据查询的场景类型获取提取信息。
        如果未找到对应类型，则返回一个默认提取信息对象。
        :param scenario_type: 要查询的场景类型。
        :return: 对应的场景提取信息。
        """
        return (
            self.mapping[scenario_type]
            if scenario_type in self.mapping
            else ScenarioExtractionInfo(subsample_ratio=self.subsample_ratio_override)
        )


def download_file_if_necessary(data_root: str, potentially_remote_path: str, verbose: bool = False) -> str:
    """
    如有必要，下载数据库文件。
    :param data_root: 数据根目录。
    :param potentially_remote_path: 文件路径，可能需要从远程下载。
    :param verbose: 是否启用详细日志。
    :return: 本地文件路径。
    """
    # 如果文件已存在本地，直接返回
    if os.path.exists(potentially_remote_path):
        return potentially_remote_path

    log_name = absolute_path_to_log_name(potentially_remote_path)
    download_name = log_name + ".db"

    # TODO: CacheStore 存在一些 bug，此处手动使用底层 store
    os.makedirs(data_root, exist_ok=True)
    local_store = LocalStore(data_root)

    if not local_store.exists(download_name):
        blob_store = BlobStoreCreator.create_nuplandb(data_root, verbose=verbose)

        # 推断远程路径
        logger.info("未找到 DB 路径。正在下载至 %s..." % download_name)
        start_time = time.time()

        remote_key = potentially_remote_path
        if not remote_key.startswith("s3://"):
            fixed_local_path = convert_legacy_nuplan_path_to_latest(potentially_remote_path)
            remote_key = infer_remote_key_from_local_path(fixed_local_path)

        content = blob_store.get(remote_key)
        local_store.put(download_name, content)
        logger.info("下载 db 文件耗时 %.2f 秒。" % (time.time() - start_time))

    return os.path.join(data_root, download_name)


def convert_legacy_nuplan_path_to_latest(legacy_path: str, nuplan_data_root: Optional[str] = None) -> str:
    """
    将旧版 nuPlan 路径格式转换为最新版。
    示例：
    - data_root: /data/sets/nuplan/
      输入: /data/sets/nuplan/nuplan-v1.1/mini/2021.09.16.15.12.03_veh-42_01037_01434.db
      输出: /data/sets/nuplan/nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db
    :param legacy_path: 需要转换的旧路径。
    :param nuplan_data_root: 自定义 nuPlan 数据根目录。若未提供则使用环境变量 NUPLAN_DATA_ROOT。
    :return: 转换后的输入路径。
    """
    # 安全检查，如果没有版本号则退出
    if legacy_path.find("nuplan-v") == -1:
        raise ValueError("nuPlan DB 路径中应包含版本号（如：nuplan-v1.1）")

    # 去除数据根目录，我们只关心后续部分
    if nuplan_data_root is None:
        nuplan_data_root = NUPLAN_DATA_ROOT
    prefix_removed = legacy_path.removeprefix(nuplan_data_root)

    # 确保路径不以 '/' 开头，避免与 NUPLAN_DATA_ROOT 的结尾斜杠冲突
    prefix_removed = prefix_removed.lstrip("/")

    # 如果路径中没有 "splits" 目录，插入该目录
    prefix_removed_path = Path(prefix_removed)
    if prefix_removed.find("splits") == -1:
        path_parts = list(prefix_removed_path.parts)
        version_directory_index = min(
            idx for idx, directory_name in enumerate(path_parts) if "nuplan-v" in directory_name
        )
        path_parts.insert(version_directory_index + 1, "splits")
        prefix_removed_path = Path("/".join(path_parts))

    return_path = Path(nuplan_data_root) / prefix_removed_path

    return str(return_path)


def infer_remote_key_from_local_path(local_path: str, nuplan_data_root: Optional[str] = None) -> str:
    """
    根据本地路径推断 S3 上的远程 key。
    示例：
    - nuplan_data_root: /data/sets/nuplan/
      输入: /data/sets/nuplan/nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db
      输出: splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db
    :param local_path: 本地文件路径。
    :param nuplan_data_root: 自定义 nuPlan 数据根目录。若未提供则使用环境变量。
    :return: 推断出的远程 key。
    """
    if nuplan_data_root is None:
        nuplan_data_root = NUPLAN_DATA_ROOT
    remote_key = local_path.removeprefix(nuplan_data_root)

    # 硥路径不以 '/' 开头
    remote_key = remote_key.lstrip("/")

    # 如果路径中没有 `splits`，添加之
    if remote_key.startswith("nuplan-v"):
        remote_key_as_path = Path(remote_key)
        remote_key_as_path = Path(*remote_key_as_path.parts[1:])
        remote_key = str(remote_key_as_path)

    return remote_key


def _process_future_trajectories_for_windowed_agents(
    log_file: str,
    tracked_objects: List[TrackedObject],
    agent_indexes: Dict[int, Dict[str, int]],
    future_trajectory_sampling: TrajectorySampling,
) -> List[TrackedObject]:
    """
    辅助方法：插值并解析窗口内 agents 的未来轨迹。
    :param log_file: 要查询的日志文件。
    :param tracked_objects: 要处理的追踪对象列表。
    :param agent_indexes: 映射 [timestamp, [track_token, tracked_object_idx]]
    :param future_trajectory_sampling: 用于未来路径点的轨迹采样。
    :return: 包含预测轨迹的追踪对象列表。
    """
    agent_future_trajectories: Dict[int, Dict[str, List[Waypoint]]] = {}
    for timestamp in agent_indexes:
        agent_future_trajectories[timestamp] = {}

        for token in agent_indexes[timestamp]:
            agent_future_trajectories[timestamp][token] = []

    for timestamp_time in agent_future_trajectories:
        end_time = timestamp_time + int(
            1e6 * (future_trajectory_sampling.time_horizon + future_trajectory_sampling.interval_length)
        )

        # TODO: 这里效率较低，建议将重采样逻辑放在 SQL 层

        for track_token, waypoint in get_future_waypoints_for_agents_from_db(
            log_file, list(agent_indexes[timestamp_time].keys()), timestamp_time, end_time
        ):
            agent_future_trajectories[timestamp_time][track_token].append(waypoint)

    for key in agent_future_trajectories:
        for track_token in agent_future_trajectories[key]:
            # 只有未来路径点多于1个才能进行插值
            if len(agent_future_trajectories[key][track_token]) == 1:
                tracked_objects[agent_indexes[key][track_token]]._predictions = [
                    PredictedTrajectory(1.0, agent_future_trajectories[key][track_token])
                ]
            elif len(agent_future_trajectories[key][track_token]) > 1:
                tracked_objects[agent_indexes[key][track_token]]._predictions = [
                    PredictedTrajectory(
                        1.0,
                        interpolate_future_waypoints(
                            agent_future_trajectories[key][track_token],
                            future_trajectory_sampling.time_horizon,
                            future_trajectory_sampling.interval_length,
                        ),
                    )
                ]

    return tracked_objects


def extract_tracked_objects_within_time_window(
    token: str,
    log_file: str,
    past_time_horizon: float,
    future_time_horizon: float,
    filter_track_tokens: Optional[Set[str]] = None,
    future_trajectory_sampling: Optional[TrajectorySampling] = None,
) -> TrackedObjects:
    """
    提取以指定 token 为中心的时间窗口内的所有追踪对象。
    :param token: 时间窗口中心 token。
    :param log_file: 日志文件路径。
    :param past_time_horizon: 向前查找的时间范围 [s]。
    :param future_time_horizon: 向后查找的时间范围 [s]。
    :param filter_track_tokens: 若提供，则只保留这些 track_token 的对象。
    :param future_trajectory_sampling: 若提供，则用于插值未来路径点。
    :return: 提取的追踪对象集合。
    """
    tracked_objects: List[TrackedObject] = []
    agent_indexes: Dict[int, Dict[str, int]] = {}

    token_timestamp = get_sensor_data_token_timestamp_from_db(log_file, get_lidarpc_sensor_data(), token)
    start_time = int(token_timestamp - (1e6 * past_time_horizon))
    end_time = int(token_timestamp + (1e6 * future_time_horizon))

    for idx, tracked_object in enumerate(
        get_tracked_objects_within_time_interval_from_db(log_file, start_time, end_time, filter_track_tokens)
    ):
        if future_trajectory_sampling and isinstance(tracked_object, Agent):
            if tracked_object.metadata.timestamp_us not in agent_indexes:
                agent_indexes[tracked_object.metadata.timestamp_us] = {}

            agent_indexes[tracked_object.metadata.timestamp_us][tracked_object.metadata.track_token] = idx
        tracked_objects.append(tracked_object)

    if future_trajectory_sampling:
        _process_future_trajectories_for_windowed_agents(
            log_file, tracked_objects, agent_indexes, future_trajectory_sampling
        )

    return TrackedObjects(tracked_objects=tracked_objects)


def extract_tracked_objects(
    token: str,
    log_file: str,
    future_trajectory_sampling: Optional[TrajectorySampling] = None,
) -> TrackedObjects:
    """
    提取 lidar_pc 中的所有检测框。
    :param token: lidar_pc 的 token。
    :param log_file: 日志文件路径。
    :param future_trajectory_sampling: 若提供，用于插值未来路径点。
    :return: lidar_pc 中包含的追踪对象。
    """
    tracked_objects: List[TrackedObject] = []
    agent_indexes: Dict[str, int] = {}
    agent_future_trajectories: Dict[str, List[Waypoint]] = {}

    for idx, tracked_object in enumerate(get_tracked_objects_for_lidarpc_token_from_db(log_file, token)):
        if future_trajectory_sampling and isinstance(tracked_object, Agent):
            agent_indexes[tracked_object.metadata.track_token] = idx
            agent_future_trajectories[tracked_object.metadata.track_token] = []
        tracked_objects.append(tracked_object)

    if future_trajectory_sampling and len(tracked_objects) > 0:
        timestamp_time = get_sensor_data_token_timestamp_from_db(log_file, get_lidarpc_sensor_data(), token)
        end_time = timestamp_time + int(
            1e6 * (future_trajectory_sampling.time_horizon + future_trajectory_sampling.interval_length)
        )

        # TODO: 此处效率较低，建议将重采样逻辑放在 SQL 层

        for track_token, waypoint in get_future_waypoints_for_agents_from_db(
            log_file, list(agent_indexes.keys()), timestamp_time, end_time
        ):
            agent_future_trajectories[track_token].append(waypoint)

        for key in agent_future_trajectories:
            # 只有未来路径点多于1个才能进行插值
            if len(agent_future_trajectories[key]) == 1:
                tracked_objects[agent_indexes[key]]._predictions = [
                    PredictedTrajectory(1.0, agent_future_trajectories[key])
                ]
            elif len(agent_future_trajectories[key]) > 1:
                tracked_objects[agent_indexes[key]]._predictions = [
                    PredictedTrajectory(
                        1.0,
                        interpolate_future_waypoints(
                            agent_future_trajectories[key],
                            future_trajectory_sampling.time_horizon,
                            future_trajectory_sampling.interval_length,
                        ),
                    )
                ]

    return TrackedObjects(tracked_objects=tracked_objects)


def extract_sensor_tokens_as_scenario(
    log_file: str,
    sensor_data_source: SensorDataSource,
    anchor_timestamp: float,
    scenario_extraction_info: ScenarioExtractionInfo,
) -> Generator[str, None, None]:
    """
    提取围绕锚定时间戳的一系列传感器 token 来组成一个场景。
    :param log_file: 要访问的日志文件。
    :param sensor_data_source: 查询目标表的参数。
    :param anchor_timestamp: 场景起始时间戳。
    :param scenario_extraction_info: 包含提取场景所需信息的结构体。
    :return: 提取到的传感器 token 列表。
    """
    start_timestamp = int(anchor_timestamp + scenario_extraction_info.extraction_offset * 1e6)
    end_timestamp = int(start_timestamp + scenario_extraction_info.scenario_duration * 1e6)
    subsample_step = int(1.0 / scenario_extraction_info.subsample_ratio)

    return cast(
        Generator[str, None, None],
        get_sampled_sensor_tokens_in_time_window_from_db(
            log_file, sensor_data_source, start_timestamp, end_timestamp, subsample_step
        ),
    )


def absolute_path_to_log_name(absolute_path: str) -> str:
    """
    从日志文件的绝对路径中提取日志名称。
    示例：
    - 输入: data/sets/nuplan/nuplan-v1.1/splits/mini/2021.10.11.02.57.41_veh-50_01522_02088.db
      输出: 2021.10.11.02.57.41_veh-50_01522_02088
    - 输入: /tmp/abcdef
      输出: abcdef
    :param absolute_path: 日志文件的绝对路径。
    :return: 日志名称。
    """
    filename = os.path.basename(absolute_path)

    # 缓存生成的文件没有 .db 扩展名
    if filename.endswith(".db"):
        filename = os.path.splitext(filename)[0]
    return filename


def download_and_cache(key: str, local_store: LocalStore, remote_store: S3Store) -> Optional[BinaryIO]:
    """
    下载并缓存指定 key 的数据。
    本函数假设本地和远程存储已经配置好。
    数据将从远程存储下载并保存在本地存储中。
    如果不存在 blob store，将初始化它。

    :param key: 要获取的数据 key。
    :param local_store: 本地 blob store。
    :param remote_store: 远程 blob store。
    :return: 获取的传感器数据。
    """
    if local_store.exists(key):
        return cast(BinaryIO, local_store.get(key))

    if remote_store is None:
        raise RuntimeError("远程存储未设置且本地未找到 key。请尝试设置 NUPLAN_DATA_STORE='s3'")

    try:
        blob = remote_store.get(key)
        local_store.put(key, blob)
        return cast(BinaryIO, local_store.get(key))
    except RuntimeError as error:
        logging.warning(f"未找到传感器数据。原因：{error}")
        return None


def load_point_cloud(lidar_pc: LidarPc, local_store: LocalStore, remote_store: S3Store) -> Optional[LidarPointCloud]:
    """
    加载点云数据。
    :param lidar_pc: LidarPC 对象。
    :param local_store: 本地 blob store。
    :param remote_store: 远程 blob store。
    :return: 加载的点云数据。
    """
    file_type = lidar_pc.filename.split('.')[-1]
    blob = download_and_cache(lidar_pc.filename, local_store, remote_store)
    return LidarPointCloud.from_buffer(blob, file_type) if blob is not None else None


def load_image(image: ImageDBRow.Image, local_store: LocalStore, remote_store: S3Store) -> Optional[Image]:
    """
    加载图像数据。
    :param image: 图像数据库行对象。
    :param local_store: 本地 blob store。
    :param remote_store: 远程 blob store。
    :return: 加载的图像数据。
    """
    blob = download_and_cache(image.filename_jpg, local_store, remote_store)
    return Image.from_buffer(blob) if blob is not None else None