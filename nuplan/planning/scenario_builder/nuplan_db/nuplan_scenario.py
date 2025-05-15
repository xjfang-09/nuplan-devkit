from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path
from typing import Any, Generator, List, Optional, Set, Tuple, Type, cast

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatuses, Transform
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.maps.nuplan_map.utils import get_roadblock_ids_from_trajectory
from nuplan.database.common.blob_store.local_store import LocalStore
from nuplan.database.common.blob_store.s3_store import S3Store
from nuplan.database.nuplan_db.lidar_pc import LidarPc
from nuplan.database.nuplan_db.nuplan_db_utils import get_lidarpc_sensor_data
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_ego_state_for_lidarpc_token_from_db,
    get_end_sensor_time_from_db,
    get_images_from_lidar_tokens,
    get_mission_goal_for_sensor_data_token_from_db,
    get_roadblock_ids_for_lidarpc_token_from_db,
    get_sampled_ego_states_from_db,
    get_sampled_lidarpcs_from_db,
    get_sensor_data_from_sensor_data_tokens_from_db,
    get_sensor_data_token_timestamp_from_db,
    get_sensor_transform_matrix_for_sensor_data_token_from_db,
    get_statese2_for_lidarpc_token_from_db,
    get_traffic_light_status_for_lidarpc_token_from_db,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    ScenarioExtractionInfo,
    absolute_path_to_log_name,
    download_file_if_necessary,
    extract_sensor_tokens_as_scenario,
    extract_tracked_objects,
    extract_tracked_objects_within_time_window,
    load_image,
    load_point_cloud,
)
from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon
from nuplan.planning.simulation.observation.observation_type import (
    CameraChannel,
    DetectionsTracks,
    LidarChannel,
    SensorChannel,
    Sensors,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class NuPlanScenario(AbstractScenario):
    """用于 nuPlan 数据集的场景实现，适用于训练与仿真模块。"""

    def __init__(
        self,
        data_root: str,
        log_file_load_path: str,
        initial_lidar_token: str,
        initial_lidar_timestamp: int,
        scenario_type: str,
        map_root: str,
        map_version: str,
        map_name: str,
        scenario_extraction_info: Optional[ScenarioExtractionInfo],
        ego_vehicle_parameters: VehicleParameters,
        sensor_root: Optional[str] = None,
    ) -> None:
        """
        初始化 nuPlan 场景。
        :param data_root: 日志文件路径前缀，如 "/data/root/nuplan"。如果是远程路径，此目录将用于下载文件。
        :param log_file_load_path: 当前场景所属日志文件名称，如 "s3://path/to/db.db" 或本地路径。
        :param initial_lidar_token: 场景初始 lidarpc 的 token。
        :param initial_lidar_timestamp: 初始 lidarpc 的时间戳。
        :param scenario_type: 场景类型（如：自车超车）。
        :param map_root: 地图数据库根路径。
        :param map_version: 使用的地图版本。
        :param map_name: 使用的地图名称。
        :param scenario_extraction_info: 包含提取场景信息的结构体。若为 None 表示仅使用初始 lidarpc。
        :param ego_vehicle_parameters: 自车车辆参数结构体。
        :param sensor_root: 传感器数据存储根路径。
        """
        # 延迟创建 blob store
        self._local_store: Optional[LocalStore] = None
        self._remote_store: Optional[S3Store] = None

        self._data_root = data_root
        self._log_file_load_path = log_file_load_path
        self._initial_lidar_token = initial_lidar_token
        self._initial_lidar_timestamp = initial_lidar_timestamp
        self._scenario_type = scenario_type
        self._map_root = map_root
        self._map_version = map_version
        self._map_name = map_name
        self._scenario_extraction_info = scenario_extraction_info
        self._ego_vehicle_parameters = ego_vehicle_parameters
        self._sensor_root = sensor_root

        # 如果提供了场景提取信息，则校验子采样率是否合法
        if self._scenario_extraction_info is not None:
            skip_rows = 1.0 / self._scenario_extraction_info.subsample_ratio
            if abs(int(skip_rows) - skip_rows) > 1e-3:
                raise ValueError(
                    f"子采样率不合法。必须是整数行跳过比率，当前值 {self._scenario_extraction_info.subsample_ratio} 会跳过 {skip_rows} 行。"
                )

        # 数据库中连续行的时间间隔（秒）
        self._database_row_interval = 0.05

        # 通常，此时日志文件已被下载，如果没有则在此处下载
        self._log_file = download_file_if_necessary(self._data_root, self._log_file_load_path)
        self._log_name: str = absolute_path_to_log_name(self._log_file)

    def __reduce__(self) -> Tuple[Type[NuPlanScenario], Tuple[Any, ...]]:
        """
        用于对象序列化时的重建提示。
        :return: 对象类型和构造参数。
        """
        return (
            self.__class__,
            (
                self._data_root,
                self._log_file_load_path,
                self._initial_lidar_token,
                self._initial_lidar_timestamp,
                self._scenario_type,
                self._map_root,
                self._map_version,
                self._map_name,
                self._scenario_extraction_info,
                self._ego_vehicle_parameters,
                self._sensor_root,
            ),
        )

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """继承自父类，请参考文档"""
        return self._ego_vehicle_parameters

    @cached_property
    def _lidarpc_tokens(self) -> List[str]:
        """
        获取该场景包含的所有 lidarpc token。
        :return: lidarpc token 字符串列表。
        """
        if self._scenario_extraction_info is None:
            return [self._initial_lidar_token]

        lidarpc_tokens = list(
            extract_sensor_tokens_as_scenario(
                self._log_file,
                get_lidarpc_sensor_data(),
                self._initial_lidar_timestamp,
                self._scenario_extraction_info,
            )
        )

        return cast(List[str], lidarpc_tokens)

    @cached_property
    def _route_roadblock_ids(self) -> List[str]:
        """
        返回专家轨迹对应的 roadblock ID 列表。
        :return: 路径上所有道路块 ID。
        """
        expert_trajectory = list(self._extract_expert_trajectory())
        return get_roadblock_ids_from_trajectory(self.map_api, expert_trajectory)  # type: ignore

    @property
    def token(self) -> str:
        """继承自父类，请参考文档"""
        return self._initial_lidar_token

    @property
    def log_name(self) -> str:
        """继承自父类，请参考文档"""
        return self._log_name

    @property
    def scenario_name(self) -> str:
        """继承自父类，请参考文档"""
        return self.token

    @property
    def scenario_type(self) -> str:
        """继承自父类，请参考文档"""
        return self._scenario_type

    @property
    def map_api(self) -> AbstractMap:
        """继承自父类，请参考文档"""
        return get_maps_api(self._map_root, self._map_version, self._map_name)

    @property
    def map_root(self) -> str:
        """获取地图根文件夹"""
        return self._map_root

    @property
    def map_version(self) -> str:
        """获取地图版本"""
        return self._map_version

    @property
    def database_interval(self) -> float:
        """继承自父类，请参考文档"""
        if self._scenario_extraction_info is None:
            return 0.05  # 默认为 20Hz
        return float(0.05 / self._scenario_extraction_info.subsample_ratio)

    def get_number_of_iterations(self) -> int:
        """继承自父类，请参考文档"""
        return len(self._lidarpc_tokens)

    def get_lidar_to_ego_transform(self) -> Transform:
        """继承自父类，请参考文档"""
        return get_sensor_transform_matrix_for_sensor_data_token_from_db(
            self._log_file, get_lidarpc_sensor_data(), self._initial_lidar_token
        )

    def get_mission_goal(self) -> Optional[StateSE2]:
        """继承自父类，请参考文档"""
        return get_mission_goal_for_sensor_data_token_from_db(
            self._log_file, get_lidarpc_sensor_data(), self._initial_lidar_token
        )

    def get_route_roadblock_ids(self) -> List[str]:
        """继承自父类，请参考文档"""
        roadblock_ids = get_roadblock_ids_for_lidarpc_token_from_db(self._log_file, self._initial_lidar_token)
        assert roadblock_ids is not None, "无法找到当前场景对应的 Roadblock ID！"
        return cast(List[str], roadblock_ids)

    def get_expert_goal_state(self) -> StateSE2:
        """继承自父类，请参考文档"""
        return get_statese2_for_lidarpc_token_from_db(self._log_file, self._lidarpc_tokens[-1])

    def get_time_point(self, iteration: int) -> TimePoint:
        """继承自父类，请参考文档"""
        return TimePoint(
            time_us=get_sensor_data_token_timestamp_from_db(
                self._log_file, get_lidarpc_sensor_data(), self._lidarpc_tokens[iteration]
            )
        )

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """继承自父类，请参考文档"""
        return get_ego_state_for_lidarpc_token_from_db(self._log_file, self._lidarpc_tokens[iteration])

    def get_tracked_objects_at_iteration(
        self,
        iteration: int,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """继承自父类，请参考文档"""
        assert 0 <= iteration < self.get_number_of_iterations(), f"迭代超出范围: {iteration}!"
        return DetectionsTracks(
            extract_tracked_objects(self._lidarpc_tokens[iteration], self._log_file, future_trajectory_sampling)
        )

    def get_tracked_objects_within_time_window_at_iteration(
        self,
        iteration: int,
        past_time_horizon: float,
        future_time_horizon: float,
        filter_track_tokens: Optional[Set[str]] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """继承自父类，请参考文档"""
        assert 0 <= iteration < self.get_number_of_iterations(), f"迭代超出范围: {iteration}!"
        return DetectionsTracks(
            extract_tracked_objects_within_time_window(
                self._lidarpc_tokens[iteration],
                self._log_file,
                past_time_horizon,
                future_time_horizon,
                filter_track_tokens,
                future_trajectory_sampling,
            )
        )

    def get_sensors_at_iteration(self, iteration: int, channels: Optional[List[SensorChannel]] = None) -> Sensors:
        """继承自父类，请参考文档"""
        # 兼容旧版本，默认返回 MERGED_PC 点云数据
        channels = [LidarChannel.MERGED_PC] if channels is None else channels

        lidar_pc = next(
            get_sensor_data_from_sensor_data_tokens_from_db(
                self._log_file, get_lidarpc_sensor_data(), LidarPc, [self._lidarpc_tokens[iteration]]
            )
        )
        return self._get_sensor_data_from_lidar_pc(cast(LidarPc, lidar_pc), channels)

    def get_future_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        """继承自父类，请参考文档"""
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True):
            yield TimePoint(lidar_pc.timestamp)

    def get_past_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        """继承自父类，请参考文档"""
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield TimePoint(lidar_pc.timestamp)

    def get_ego_past_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        """继承自父类，请参考文档"""
        num_samples = num_samples if num_samples else int(time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._database_row_interval)

        return cast(
            Generator[EgoState, None, None],
            get_sampled_ego_states_from_db(
                self._log_file, self._lidarpc_tokens[iteration], get_lidarpc_sensor_data(), indices, future=False
            ),
        )

    def get_ego_future_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        """继承自父类，请参考文档"""
        num_samples = num_samples if num_samples else int(time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._database_row_interval)

        return cast(
            Generator[EgoState, None, None],
            get_sampled_ego_states_from_db(
                self._log_file, self._lidarpc_tokens[iteration], get_lidarpc_sensor_data(), indices, future=True
            ),
        )

    def get_past_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        """继承自父类，请参考文档"""
        # TODO: 可以通过批量查询进一步优化性能
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield DetectionsTracks(extract_tracked_objects(lidar_pc.token, self._log_file, future_trajectory_sampling))

    def get_future_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        """继承自父类，请参考文档"""
        # TODO: 可以通过批量查询进一步优化性能
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True):
            yield DetectionsTracks(extract_tracked_objects(lidar_pc.token, self._log_file, future_trajectory_sampling))

    def get_past_sensors(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        channels: Optional[List[SensorChannel]] = None,
    ) -> Generator[Sensors, None, None]:
        """继承自父类，请参考文档"""
        # 兼容旧版本，默认返回点云数据
        channels = [LidarChannel.MERGED_PC] if channels is None else channels

        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield self._get_sensor_data_from_lidar_pc(lidar_pc, channels)

    def get_traffic_light_status_at_iteration(self, iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        """继承自父类，请参考文档"""
        token = self._lidarpc_tokens[iteration]

        return cast(
            Generator[TrafficLightStatusData, None, None],
            get_traffic_light_status_for_lidarpc_token_from_db(self._log_file, token),
        )

    def get_past_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        """
        获取过去时间段内的交通灯状态历史。
        
        :param iteration: 场景中的迭代次数，0 <= iteration < 总帧数。
        :param time_horizon: [s] 查找过去状态的时间窗口。
        :param num_samples: {num_samples}条目数，如果未提供则从 DB 推断。
        :return: 过去交通灯状态生成器。
        """
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield TrafficLightStatuses(
                list(get_traffic_light_status_for_lidarpc_token_from_db(self._log_file, lidar_pc.token))
            )

    def get_future_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        """
        获取未来时间段内的交通灯状态历史。

        :param iteration: 场景中的迭代次数，0 <= iteration < 总帧数。
        :param time_horizon: [s] 查找未来状态的时间窗口。
        :param num_samples: 要获取的数据条目数。
        :return: 未来交通灯状态生成器。
        """
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True):
            yield TrafficLightStatuses(
                list(get_traffic_light_status_for_lidarpc_token_from_db(self._log_file, lidar_pc.token))
            )

    def get_scenario_tokens(self) -> List[str]:
        """返回该场景中所有的 lidarpc token 列表"""
        return self._lidarpc_tokens

    def _find_matching_lidar_pcs(
        self, iteration: int, num_samples: Optional[int], time_horizon: float, look_into_future: bool
    ) -> Generator[LidarPc, None, None]:
        """
        根据指定时间窗口查找最匹配的 lidar_pc 数据。
        :param iteration: 场景内迭代次数。
        :param num_samples: 要获取的条目数，若未提供则从 DB 推断。
        :param time_horizon: [s] 查找时间窗口。
        :param look_into_future: 如果为 True 表示向前查找；否则表示向后查找。
        :return: 匹配到的 lidar_pc 数据生成器。
        """
        num_samples = num_samples if num_samples else int(time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._database_row_interval)

        return cast(
            Generator[LidarPc, None, None],
            get_sampled_lidarpcs_from_db(
                self._log_file, self._lidarpc_tokens[iteration], get_lidarpc_sensor_data(), indices, look_into_future
            ),
        )

    def _extract_expert_trajectory(self, max_future_seconds: int = 60) -> Generator[EgoState, None, None]:
        """
        提取专家驾驶轨迹，根据给定时间参数。
        如果初始 lidar_pc 没有足够历史或未来数据，只提取可用数据。
        
        :param max_future_seconds: [s] 提取路径的最大未来时间。
        :return: 专家驾驶轨迹的 EgoState 列表。
        """
        minimal_required_future_time_available = 0.5

        # 提取未来数据
        end_log_time_us = get_end_sensor_time_from_db(self._log_file, get_lidarpc_sensor_data())
        max_future_time = min((end_log_time_us - self._initial_lidar_timestamp) * 1e-6, max_future_seconds)

        if max_future_time < minimal_required_future_time_available:
            return

        for traj in self.get_ego_future_trajectory(0, max_future_time):
            yield traj

    def _create_blob_store_if_needed(self) -> Tuple[LocalStore, Optional[S3Store]]:
        """
        创建 blob 存储（如不存在）。
        :return: 创建或已缓存的 LocalStore 和 S3Store 实例。
        """
        if self._local_store is not None and self._remote_store is not None:
            return self._local_store, self._remote_store

        if self._sensor_root is None:
            raise ValueError("未设置 sensor_root，请在初始化时提供 sensor_root 来访问传感器数据。")
        Path(self._sensor_root).mkdir(exist_ok=True)
        self._local_store = LocalStore(self._sensor_root)
        if os.getenv("NUPLAN_DATA_STORE", "") == "s3":
            s3_url = os.getenv("NUPLAN_DATA_ROOT_S3_URL", "")
            self._remote_store = S3Store(os.path.join(s3_url, "sensor_blobs"), show_progress=True)

        return self._local_store, self._remote_store

    def _get_sensor_data_from_lidar_pc(self, lidar_pc: LidarPc, channels: List[SensorChannel]) -> Sensors:
        """
        给定数据库中的 LidarPC 对象，加载对应的传感器数据。
        :param lidar_pc: LidarPC 数据对象。
        :param channels: 要加载的传感器通道。
        :return: 加载完成的传感器数据。
        """
        local_store, remote_store = self._create_blob_store_if_needed()

        retrieved_images = get_images_from_lidar_tokens(
            self._log_file, [lidar_pc.token], [cast(str, channel.value) for channel in channels]
        )
        lidar_pcs = (
            {LidarChannel.MERGED_PC: load_point_cloud(cast(LidarPc, lidar_pc), local_store, remote_store)}
            if LidarChannel.MERGED_PC in channels
            else None
        )

        images = {
            CameraChannel[image.channel]: load_image(image, local_store, remote_store) for image in retrieved_images
        }

        return Sensors(pointcloud=lidar_pcs, images=images if images else None)