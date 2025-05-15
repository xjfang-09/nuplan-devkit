import pickle
import sqlite3
from typing import Generator, List, Optional, Set, Tuple, Type, Union

import numpy as np
from pyquaternion import Quaternion

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType, Transform
from nuplan.common.utils.helpers import get_unique_incremental_track_id
from nuplan.database.nuplan_db.camera import Camera
from nuplan.database.nuplan_db.image import Image
from nuplan.database.nuplan_db.lidar_pc import LidarPc
from nuplan.database.nuplan_db.nuplan_db_utils import SensorDataSource
from nuplan.database.nuplan_db.query_session import execute_many, execute_one
from nuplan.database.nuplan_db.sensor_data_table_row import SensorDataTableRow
from nuplan.database.utils.label.utils import local2agent_type, raw_mapping


def _parse_tracked_object_row(row: sqlite3.Row) -> TrackedObject:
    """
    从 sqlite3 行解析一个 TrackedObject。
    :param row: 数据库查询返回的一行数据。
    :return: 解析后的 TrackedObject。
    """
    category_name = row["category_name"]
    pose = StateSE2(row["x"], row["y"], row["yaw"])
    oriented_box = OrientedBox(pose, width=row["width"], length=row["length"], height=row["height"])

    label_local = raw_mapping["global2local"][category_name]
    tracked_object_type = TrackedObjectType[local2agent_type[label_local]]

    if tracked_object_type in AGENT_TYPES:
        return Agent(
            tracked_object_type=tracked_object_type,
            oriented_box=oriented_box,
            velocity=StateVector2D(row["vx"], row["vy"]),
            predictions=[],  # 后续填充
            angular_velocity=np.nan,
            metadata=SceneObjectMetadata(
                token=row["token"].hex(),
                track_token=row["track_token"].hex(),
                track_id=get_unique_incremental_track_id(str(row["track_token"].hex())),
                timestamp_us=row["timestamp"],
                category_name=category_name,
            ),
        )
    else:
        return StaticObject(
            tracked_object_type=tracked_object_type,
            oriented_box=oriented_box,
            metadata=SceneObjectMetadata(
                token=row["token"].hex(),
                track_token=row["track_token"].hex(),
                track_id=get_unique_incremental_track_id(str(row["track_token"].hex())),
                timestamp_us=row["timestamp"],
                category_name=category_name,
            ),
        )


def get_sensor_token_by_index_from_db(log_file: str, sensor_source: SensorDataSource, index: int) -> Optional[str]:
    """
    根据时间戳顺序获取第 N 个传感器 token。
    主要用于单元测试。如果索引不存在（如 index=10000 但只有 1000 条记录），则返回 None。
    只支持非负整数索引。

    :param log_file: 要查询的数据库文件。
    :param sensor_source: 查询目标表的参数。
    :param index: lidarpc token 的 0-indexed 索引。
    :return: 如果存在则返回 token。
    """
    if index < 0:
        raise ValueError(f"传入了负索引 {index} 给 get_lidarpc_token_by_index_from_db()。")

    sensor_token = get_sensor_token(log_file, sensor_source.sensor_table, sensor_source.channel)

    query = f"""
    WITH ordered AS
    (
        SELECT  token,
                lidar_token,
                ROW_NUMBER() OVER (ORDER BY timestamp ASC) AS row_num
        FROM {sensor_source.table}
    )
    SELECT token
    FROM ordered
    WHERE (row_num - 1) = ?
        AND {sensor_source.sensor_token_column} = ?;
    """

    result = execute_one(query, [index, bytearray.fromhex(sensor_token)], log_file)
    return None if result is None else str(result["token"].hex())


def get_end_sensor_time_from_db(log_file: str, sensor_source: SensorDataSource) -> int:
    """
    获取日志文件中最后一条传感器数据的时间戳。
    :param log_file: 要查询的数据库文件。
    :param sensor_source: 查询目标表的参数。
    :return: 最后一条传感器数据的时间戳。
    """
    query = f"""
    SELECT MAX(timestamp) AS max_time
    FROM {sensor_source.table};
    """

    result = execute_one(query, [], log_file)
    return int(result["max_time"])


def get_sensor_data_token_timestamp_from_db(
    log_file: str, sensor_source: SensorDataSource, token: str
) -> Optional[int]:
    """
    获取指定 lidar_pc token 对应的时间戳。
    :param log_file: 要查询的数据库文件。
    :param sensor_source: 查询目标表的参数。
    :param token: 要查询时间戳的 token。
    :return: 如果找到则返回对应时间戳。
    """
    query = f"""
    SELECT timestamp
    FROM {sensor_source.table}
    WHERE token = ?;
    """
    result = execute_one(query, (bytearray.fromhex(token),), log_file)
    return None if result is None else int(result["timestamp"])


def get_sensor_token_map_name_from_db(log_file: str, sensor_source: SensorDataSource, token: str) -> Optional[str]:
    """
    获取指定传感器 token 所属的地图名称。
    :param log_file: 要查询的数据库文件。
    :param sensor_source: 查询目标表的参数。
    :param token: 要查询地图名称的 token。
    :return: 如果找到则返回地图名称。
    """
    query = f"""
    SELECT map_version
    FROM log AS l
    INNER JOIN {sensor_source.sensor_table} AS sensor
        ON sensor.log_token = l.token
    INNER JOIN {sensor_source.table} AS sensor_data
        ON sensor_data.{sensor_source.sensor_token_column} = sensor.token
    WHERE sensor_data.token = ?;
    """

    result = execute_one(query, (bytearray.fromhex(token),), log_file)
    return None if result is None else result["map_version"]


def get_sampled_sensor_tokens_in_time_window_from_db(
    log_file: str, sensor_source: SensorDataSource, start_timestamp: int, end_timestamp: int, subsample_interval: int
) -> Generator[str, None, None]:
    """
    在给定时间窗口 [start_timestamp, end_timestamp] 内，按 subsample_interval 间隔采样传感器 token。
    返回的 token 按时间升序排列。

    示例：
    ```
    token | timestamp
    -----------------
    1     | 0
    2     | 1
    3     | 2
    4     | 3
    5     | 4
    6     | 5
    7     | 6
    ```

    若调用时 start_timestamp=1, end_timestamp=5, subsample_interval=2，
    将返回 tokens [1, 3, 5]。

    :param log_file: 要查询的数据库文件。
    :param sensor_source: 查询目标表的参数。
    :param start_timestamp: 时间窗口起始时间（包含）。
    :param end_timestamp: 时间窗口结束时间（包含）。
    :param subsample_interval: 采样间隔。
    :return: 符合条件的 lidar_pc token 生成器。
    """
    sensor_token = get_sensor_token(log_file, sensor_source.sensor_table, sensor_source.channel)

    query = f"""
    WITH numbered AS
    (
        SELECT token, timestamp, ROW_NUMBER() OVER (ORDER BY timestamp ASC) AS row_num
        FROM {sensor_source.table}
        WHERE timestamp >= ?
        AND timestamp <= ?
        AND {sensor_source.sensor_token_column} == ?
    )
    SELECT token
    FROM numbered
    WHERE ((row_num - 1) % ?) = 0
    ORDER BY timestamp ASC;
    """

    for row in execute_many(
        query, (start_timestamp, end_timestamp, bytearray.fromhex(sensor_token), subsample_interval), log_file
    ):
        yield row["token"].hex()


def get_sensor_data_from_sensor_data_tokens_from_db(
    log_file: str,
    sensor_source: SensorDataSource,
    sensor_class: Type[SensorDataTableRow],
    tokens: Union[Generator[str, None, None], List[str]],
) -> Generator[SensorDataTableRow, None, None]:
    """
    给定一组传感器 token，构建对应的 sensor_class 实例。
    此函数对返回值排序无要求。
    
    :param sensor_source: 查询目标表的参数。
    :param sensor_class: SensorData 表中的行类。
    :param log_file: 要查询的数据库文件。
    :param tokens: 要构建对象的 token 列表或生成器。
    :return: sensor_class 对象的生成器。
    """
    if not isinstance(tokens, list):
        tokens = list(tokens)

    query = f"""
        SELECT *
        FROM {sensor_source.table}
        WHERE token IN ({('?,'*len(tokens))[:-1]});
    """

    for row in execute_many(query, [bytearray.fromhex(t) for t in tokens], log_file):
        yield sensor_class.from_db_row(row)


def get_sensor_transform_matrix_for_sensor_data_token_from_db(
    log_file: str, sensor_source: SensorDataSource, sensor_data_token: str
) -> Optional[Transform]:
    """
    获取指定 lidarpc_token 对应的变换矩阵。
    :param log_file: 要查询的日志文件。
    :param sensor_source: 查询目标表的参数。
    :param sensor_data_token: 要查询的传感器数据 token。
    :return: 变换矩阵；若不存在则返回 None。
    """
    query = f"""
        SELECT  sensor.translation,
                sensor.rotation
        FROM {sensor_source.sensor_table} AS sensor
        INNER JOIN {sensor_source.table} AS sensor_data
            ON sensor_data.{sensor_source.sensor_token_column} = sensor.token
        WHERE sensor_data.token = ?;
    """

    row = execute_one(query, (bytearray.fromhex(sensor_data_token),), log_file)
    if row is None:
        return None

    translation = pickle.loads(row["translation"])
    rotation = pickle.loads(row["rotation"])

    output = Quaternion(rotation).transformation_matrix
    output[:3, 3] = np.array(translation)

    return output


def get_mission_goal_for_sensor_data_token_from_db(
    log_file: str, sensor_source: SensorDataSource, token: str
) -> Optional[StateSE2]:
    """
    获取指定 lidar_pc token 对应的目标姿态。
    :param log_file: 要查询的日志文件。
    :param sensor_source: 查询目标表的参数。
    :param token: 要查询目标状态的 token。
    :return: 目标状态。
    """
    query = f"""
        SELECT  ep.x,
                ep.y,
                ep.qw,
                ep.qx,
                ep.qy,
                ep.qz
        FROM ego_pose AS ep
        INNER JOIN scene AS s
            ON s.goal_ego_pose_token = ep.token
        INNER JOIN {sensor_source.table} AS sensor_data
            ON sensor_data.scene_token = s.token
        WHERE sensor_data.token = ?
    """

    row = execute_one(query, (bytearray.fromhex(token),), log_file)
    if row is None:
        return None

    q = Quaternion(row["qw"], row["qx"], row["qy"], row["qz"])
    return StateSE2(row["x"], row["y"], q.yaw_pitch_roll[0])


def get_roadblock_ids_for_lidarpc_token_from_db(log_file: str, lidarpc_token: str) -> Optional[List[str]]:
    """
    获取指定 lidar_pc token 对应的场景 roadblock IDs。
    :param log_file: 要查询的数据库文件。
    :param lidarpc_token: 要查询当前状态的 token。
    :return: 字符串格式的 roadblock ID 列表。
    """
    query = """
        SELECT  s.roadblock_ids
        FROM scene AS s
        INNER JOIN lidar_pc AS lp
            ON lp.scene_token = s.token
        WHERE lp.token = ?
    """
    row = execute_one(query, (bytearray.fromhex(lidarpc_token),), log_file)
    if row is None:
        return None
    return str(row["roadblock_ids"]).split(" ")


def get_statese2_for_lidarpc_token_from_db(log_file: str, token: str) -> Optional[StateSE2]:
    """
    从数据库中根据 lidar_pc token 获取自车姿态（StateSE2）。
    :param log_file: 要查询的数据库文件。
    :param token: 要查询当前状态的 token。
    :return: 当前自车状态（StateSE2）。
    """
    query = """
        SELECT  ep.x,
                ep.y,
                ep.qw,
                ep.qx,
                ep.qy,
                ep.qz
        FROM ego_pose AS ep
        INNER JOIN lidar_pc AS lp
            ON lp.ego_pose_token = ep.token
        WHERE lp.token = ?
    """

    row = execute_one(query, (bytearray.fromhex(token),), log_file)
    if row is None:
        return None

    q = Quaternion(row["qw"], row["qx"], row["qy"], row["qz"])
    return StateSE2(row["x"], row["y"], q.yaw_pitch_roll[0])


def get_sampled_lidarpcs_from_db(
    log_file: str,
    initial_token: str,
    sensor_source: SensorDataSource,
    sample_indexes: Union[Generator[int, None, None], List[int]],
    future: bool,
) -> Generator[LidarPc, None, None]:
    """
    给定一个初始 token，返回过去或未来的 token，并按照提供的索引进行采样。
    结果始终按时间戳升序排列。

    示例：
    token | timestamp
    -----------------
    0     | 0
    1     | 1
    2     | 2
    3     | 3
    4     | 4
    5     | 5
    6     | 6
    7     | 7
    8     | 8
    9     | 9
    10    | 10

    示例结果：
    initial token | sample_indexes | future | returned tokens
    ---------------------------------------------------------
    5             | [0, 1, 2]      | True   | [5, 6, 7]
    5             | [0, 1, 2]      | False  | [3, 4, 5]
    7             | [0, 3, 12]     | False  | [4, 7]
    0             | [11]           | True   | []

    :param log_file: 要查询的数据库文件。
    :param initial_token: 查询基准 token。
    :param sample_indexes: 采样索引列表。
    :param future: 如果为 True，则表示未来 token；否则为过去 token。
    :return: 请求的 LidarPC 对象生成器。
    """
    if not isinstance(sample_indexes, list):
        sample_indexes = list(sample_indexes)

    sensor_token = get_sensor_token(log_file, sensor_source.sensor_table, sensor_source.channel)

    order_direction = "ASC" if future else "DESC"
    order_cmp = ">=" if future else "<="

    query = f"""
        WITH initial_lidarpc AS
        (
            SELECT token, timestamp
            FROM lidar_pc
            WHERE token = ?
        ),
        ordered AS
        (
            SELECT  lp.token,
                    lp.next_token,
                    lp.prev_token,
                    lp.ego_pose_token,
                    lp.lidar_token,
                    lp.scene_token,
                    lp.filename,
                    lp.timestamp,
                    ROW_NUMBER() OVER (ORDER BY lp.timestamp {order_direction}) AS row_num
            FROM lidar_pc AS lp
            CROSS JOIN initial_lidarpc AS il
            WHERE   lp.timestamp {order_cmp} il.timestamp
            AND lidar_token = ?
        )
        SELECT  token,
                next_token,
                prev_token,
                ego_pose_token,
                lidar_token,
                scene_token,
                filename,
                timestamp
        FROM ordered
        WHERE (row_num - 1) IN ({('?,'*len(sample_indexes))[:-1]})
        ORDER BY timestamp ASC;
    """

    args = [bytearray.fromhex(initial_token), bytearray.fromhex(sensor_token)] + sample_indexes  # type: ignore
    for row in execute_many(query, args, log_file):
        yield LidarPc.from_db_row(row)


def get_sampled_ego_states_from_db(
    log_file: str,
    initial_token: str,
    sensor_source: SensorDataSource,
    sample_indexes: Union[Generator[int, None, None], List[int]],
    future: bool,
) -> Generator[EgoState, None, None]:
    """
    给定一个初始 token，返回按时间排序的自车状态，按提供索引采样。
    结果始终按时间戳升序排列。

    示例：同上。

    :param log_file: 要查询的数据库文件。
    :param initial_token: 查询基准 token。
    :param sample_indexes: 采样索引。
    :param future: 如果为 True 表示未来，否则表示过去。
    :return: 自车状态生成器。
    """
    if not isinstance(sample_indexes, list):
        sample_indexes = list(sample_indexes)

    sensor_token = get_sensor_token(log_file, sensor_source.sensor_table, sensor_source.channel)

    order_direction = "ASC" if future else "DESC"
    order_cmp = ">=" if future else "<="

    query = f"""
        WITH initial_lidarpc AS
        (
            SELECT token, timestamp
            FROM lidar_pc
            WHERE token = ?
        ),
        ordered AS
        (
            SELECT  lp.token,
                    lp.next_token,
                    lp.prev_token,
                    lp.ego_pose_token,
                    lp.lidar_token,
                    lp.scene_token,
                    lp.filename,
                    lp.timestamp,
                    ROW_NUMBER() OVER (ORDER BY lp.timestamp {order_direction}) AS row_num
            FROM lidar_pc AS lp
            CROSS JOIN initial_lidarpc AS il
            WHERE   lp.timestamp {order_cmp} il.timestamp
            AND lidar_token = ?
        )
        SELECT  ep.x,
                ep.y,
                ep.qw,
                ep.qx,
                ep.qy,
                ep.qz,
                o.timestamp,
                ep.vx,
                ep.vy,
                ep.acceleration_x,
                ep.acceleration_y
        FROM ego_pose AS ep
        INNER JOIN ordered AS o
            ON o.ego_pose_token = ep.token
        WHERE (o.row_num - 1) IN ({('?,'*len(sample_indexes))[:-1]})
        ORDER BY o.timestamp ASC;
    """

    args = [bytearray.fromhex(initial_token), bytearray.fromhex(sensor_token)] + sample_indexes  # type: ignore
    for row in execute_many(query, args, log_file):
        q = Quaternion(row["qw"], row["qx"], row["qy"], row["qz"])
        yield EgoState.build_from_rear_axle(
            StateSE2(row["x"], row["y"], q.yaw_pitch_roll[0]),
            tire_steering_angle=0.0,
            vehicle_parameters=get_pacifica_parameters(),
            time_point=TimePoint(row["timestamp"]),
            rear_axle_velocity_2d=StateVector2D(row["vx"], y=row["vy"]),
            rear_axle_acceleration_2d=StateVector2D(x=row["acceleration_x"], y=row["acceleration_y"]),
        )


def get_ego_state_for_lidarpc_token_from_db(log_file: str, token: str) -> EgoState:
    """
    从数据库中获取与指定 lidar_pc token 关联的自车状态。

    :param log_file: 要查询的日志文件。
    :param token: 要查询的 lidar_pc token。
    :return: 与 LidarPC 关联的 EgoState。
    """
    query = """
        SELECT  ep.x,
                ep.y,
                ep.qw,
                ep.qx,
                ep.qy,
                ep.qz,
                lp.timestamp,
                ep.vx,
                ep.vy,
                ep.acceleration_x,
                ep.acceleration_y
        FROM ego_pose AS ep
        INNER JOIN lidar_pc AS lp
            ON lp.ego_pose_token = ep.token
        WHERE lp.token = ?
    """

    row = execute_one(query, (bytearray.fromhex(token),), log_file)
    if row is None:
        return None

    q = Quaternion(row["qw"], row["qx"], row["qy"], row["qz"])
    return EgoState.build_from_rear_axle(
        StateSE2(row["x"], row["y"], q.yaw_pitch_roll[0]),
        tire_steering_angle=0.0,
        vehicle_parameters=get_pacifica_parameters(),
        time_point=TimePoint(row["timestamp"]),
        rear_axle_velocity_2d=StateVector2D(row["vx"], y=row["vy"]),
        rear_axle_acceleration_2d=StateVector2D(x=row["acceleration_x"], y=row["acceleration_y"]),
    )


def get_traffic_light_status_for_lidarpc_token_from_db(
    log_file: str, token: str
) -> Generator[TrafficLightStatusData, None, None]:
    """
    获取与指定 lidar_pc token 关联的交通灯状态。
    :param log_file: 要查询的日志文件。
    :param token: lidar_pc token。
    :return: 与 lidar_pc 关联的交通灯状态数据。
    """
    query = """
        SELECT  CASE WHEN tl.status == "green" THEN 0
                     WHEN tl.status == "yellow" THEN 1
                     WHEN tl.status == "red" THEN 2
                     ELSE 3
                END AS status,
                tl.lane_connector_id,
                lp.timestamp AS timestamp
        FROM lidar_pc AS lp
        INNER JOIN traffic_light_status AS tl
            ON lp.token = tl.lidar_pc_token
        WHERE lp.token = ?
    """

    for row in execute_many(query, (bytearray.fromhex(token),), log_file):
        yield TrafficLightStatusData(
            status=TrafficLightStatusType(row["status"]),
            lane_connector_id=row["lane_connector_id"],
            timestamp=row["timestamp"],
        )


def get_tracked_objects_within_time_interval_from_db(
    log_file: str, start_timestamp: int, end_timestamp: int, filter_track_tokens: Optional[Set[str]] = None
) -> Generator[TrackedObject, None, None]:
    """
    获取在指定时间区间内的所有追踪对象（包括 agent 和 static object）。
    可选地根据 track_token 进行过滤。

    不会获取 agent 的未来路径点。
    如需未来路径点，请调用 [get_future_waypoints_for_agents_from_db()]
    (file:///home/mark/nuplan-devkit/nuplan/database/nuplan_db/nuplan_scenario_queries.py#L732-L777)。

    :param log_file: {log_file} 文件。
    :param start_timestamp: 查询开始时间戳 [us]。
    :param end_timestamp: 查询结束时间戳 [us]。
    :param filter_track_tokens: 如果提供，则只返回这些 track_token 的对象。
    :return: TrackedObjects 生成器，按时间戳和 track_token 排序。
    """

    args: List[Union[int, bytearray]] = [start_timestamp, end_timestamp]

    filter_clause = ""
    if filter_track_tokens is not None:
        filter_clause = """
            AND lb.track_token IN ({('?,'*len(filter_track_tokens))[:-1]})
        """
        for token in filter_track_tokens:
            args.append(bytearray.fromhex(token))

    query = f"""
        SELECT  c.name AS category_name,
                lb.x,
                lb.y,
                lb.z,
                lb.yaw,
                lb.width,
                lb.length,
                lb.height,
                lb.vx,
                lb.vy,
                lb.token,
                lb.track_token,
                lp.timestamp
        FROM lidar_box AS lb
        INNER JOIN track AS t
            ON t.token = lb.track_token
        INNER JOIN category AS c
            ON c.token = t.category_token
        INNER JOIN lidar_pc AS lp
            ON lp.token = lb.lidar_pc_token
        WHERE lp.timestamp >= ?
            AND lp.timestamp <= ?
            {filter_clause}
        ORDER BY lp.timestamp ASC, lb.track_token ASC;
    """
    for row in execute_many(query, args, log_file):
        yield _parse_tracked_object_row(row)


def get_tracked_objects_for_lidarpc_token_from_db(log_file: str, token: str) -> Generator[TrackedObject, None, None]:
    """
    获取指定 lidar_pc 的所有追踪对象。
    包括 agent 和静态对象。
    返回结果是无序的。

    对于 agent，此查询不会获取未来路径点。
    如需未来路径点，请调用 `get_future_waypoints_for_agents_from_db()` 并传入对应的 agent track_tokens。

    :param log_file: 要查询的日志文件。
    :param token: 用于查询追踪对象的 lidar_pc token。
    :return: 与该 token 关联的 TrackedObjects。
    """
    query = """
        SELECT  c.name AS category_name,
                lb.x,
                lb.y,
                lb.z,
                lb.yaw,
                lb.width,
                lb.length,
                lb.height,
                lb.vx,
                lb.vy,
                lb.token,
                lb.track_token,
                lp.timestamp
        FROM lidar_box AS lb
        INNER JOIN track AS t
            ON t.token = lb.track_token
        INNER JOIN category AS c
            ON c.token = t.category_token
        INNER JOIN lidar_pc AS lp
            ON lp.token = lb.lidar_pc_token
        WHERE lp.token = ?
    """

    for row in execute_many(query, (bytearray.fromhex(token),), log_file):
        yield _parse_tracked_object_row(row)


def get_future_waypoints_for_agents_from_db(
    log_file: str, track_tokens: Union[Generator[str, None, None], List[str]], start_timestamp: int, end_timestamp: int
) -> Generator[Tuple[str, Waypoint], None, None]:
    """
    获取在指定时间窗口内 agent 的未来路径点。
    结果按 track_token 升序、timestamp 升序排列。

    :param log_file: 要查询的日志文件。
    :param track_tokens: 需要查询的 track_token 列表或生成器。
    :param start_timestamp: 查询起始时间戳 [us]。
    :param end_timestamp: 查询结束时间戳 [us]。
    :return: (track_token, Waypoint) 元组的生成器，按 track_token 和 timestamp 排序。
    """
    if not isinstance(track_tokens, list):
        track_tokens = list(track_tokens)

    query = f"""
        SELECT  lb.x,
                lb.y,
                lb.z,
                lb.yaw,
                lb.width,
                lb.length,
                lb.height,
                lb.vx,
                lb.vy,
                lb.track_token,
                lp.timestamp
        FROM lidar_box AS lb
        INNER JOIN lidar_pc AS lp
            ON lp.token = lb.lidar_pc_token
        WHERE   lp.timestamp >= ?
            AND lp.timestamp <= ?
            AND lb.track_token IN
            ({('?,'*len(track_tokens))[:-1]})
        ORDER BY lb.track_token ASC, lp.timestamp ASC;
    """

    args = [start_timestamp, end_timestamp] + [bytearray.fromhex(t) for t in track_tokens]  # type: ignore

    for row in execute_many(query, args, log_file):
        pose = StateSE2(row["x"], row["y"], row["yaw"])
        oriented_box = OrientedBox(pose, width=row["width"], height=row["height"], length=row["length"])
        velocity = StateVector2D(row["vx"], row["vy"])

        yield (row["track_token"].hex(), Waypoint(TimePoint(row["timestamp"]), oriented_box, velocity))


def get_scenarios_from_db(
    log_file: str,
    filter_tokens: Optional[List[str]],
    filter_types: Optional[List[str]],
    filter_map_names: Optional[List[str]],
    include_invalid_mission_goals: bool = True,
    include_cameras: bool = False,
) -> Generator[sqlite3.Row, None, None]:
    """
    获取符合筛选条件的场景信息。
    如果某个筛选条件为 None，则不加入查询过滤。
    返回结果按时间戳升序排序。

    :param log_file: 要查询的日志文件。
    :param filter_tokens: 若提供，只返回这些 token 的场景。
    :param filter_types: 若提供，只返回这些类型的场景。
    :param filter_map_names: 若提供，只返回对应地图名称的场景。
    :param include_invalid_mission_goals: 如果为 True，包含没有有效目标的任务场景；
                                         如果为 False，将过滤掉无效任务目标。
    :param include_cameras: 如果为 True，只返回有图像数据的场景。
    :return: sqlite3.Row 对象，包含以下字段：
        * token: 场景初始帧的 lidar_pc token。
        * timestamp: 场景初始帧的时间戳。
        * map_name: 场景对应的地图名称。
        * scenario_type: 场景类型，可能为 None（如果没有匹配）。
    """
    filter_clauses = []
    args: List[Union[str, bytearray]] = []
    if filter_types is not None:
        filter_clauses.append(
            f"""
        st.type IN ({('?,'*len(filter_types))[:-1]})
        """
        )
        args += filter_types

    if filter_tokens is not None:
        filter_clauses.append(
            f"""
        lp.token IN ({('?,'*len(filter_tokens))[:-1]})
        """
        )
        args += [bytearray.fromhex(t) for t in filter_tokens]

    if filter_map_names is not None:
        filter_clauses.append(
            f"""
        l.map_version IN ({('?,'*len(filter_map_names))[:-1]})
        """
        )
        args += filter_map_names

    if len(filter_clauses) > 0:
        filter_clause = "WHERE " + " AND ".join(filter_clauses)
    else:
        filter_clause = ""

    if include_invalid_mission_goals:
        invalid_goals_joins = ""
    else:
        invalid_goals_joins = """
        -- 过滤掉没有有效 mission goal 的场景
        INNER JOIN scene AS invalid_goal_scene
            ON invalid_goal_scene.token = lp.scene_token
        INNER JOIN ego_pose AS invalid_goal_ego_pose
            ON invalid_goal_scene.goal_ego_pose_token = invalid_goal_ego_pose.token
        """

    if include_cameras:
        matching_camera_clause = """
        INNER JOIN image AS img
            ON img.ego_pose_token = lp.ego_pose_token
        """
    else:
        matching_camera_clause = ""

    query = f"""
        WITH ordered_scenes AS
        (
            SELECT  token,
                    ROW_NUMBER() OVER (ORDER BY name ASC) AS row_num
            FROM scene
        ),
        num_scenes AS
        (
            SELECT  COUNT(*) AS cnt
            FROM scene
        ),
        valid_scenes AS
        (
            SELECT  o.token
            FROM ordered_scenes AS o
            CROSS JOIN num_scenes AS n

            -- 定义“有效”场景：至少前后各有两个 lidar_pc 数据
            WHERE o.row_num >= 3 AND o.row_num < n.cnt - 1
        )
        SELECT  lp.token,
                lp.timestamp,
                l.map_version AS map_name,

                -- 场景可以有多个标签
                -- 此处从可用标签中任选一个作为输出
                MAX(st.type) AS scenario_type
        FROM lidar_pc AS lp
        LEFT OUTER JOIN scenario_tag AS st
            ON lp.token = st.lidar_pc_token
        INNER JOIN lidar AS ld
            ON ld.token = lp.lidar_token
        INNER JOIN log AS l
            ON ld.log_token = l.token
        INNER JOIN valid_scenes AS vs
            ON lp.scene_token = vs.token
        {matching_camera_clause}
        {invalid_goals_joins}
        {filter_clause}
        GROUP BY    lp.token,
                    lp.timestamp,
                    l.map_version
        ORDER BY lp.timestamp ASC;
    """

    for row in execute_many(query, args, log_file):
        yield row


def get_lidarpc_tokens_with_scenario_tag_from_db(log_file: str) -> Generator[Tuple[str, str], None, None]:
    """
    获取所有被标记过场景类型的 lidar_pc token。
    按场景类型升序返回。

    :param log_file: 要查询的日志文件。
    :return: (scenario_tag, token) 元组生成器。
    """
    query = """
    SELECT  st.type,
            lp.token
    FROM lidar_pc AS lp
    LEFT OUTER JOIN scenario_tag AS st
        ON lp.token=st.lidar_pc_token
    WHERE st.type IS NOT NULL
    ORDER BY st.type ASC NULLS LAST;
    """

    for row in execute_many(query, (), log_file):
        yield (str(row["type"]), row["token"].hex())


def get_sensor_token(log_file: str, table: str, channel: str) -> str:
    """
    获取指定传感器通道的 token。
    :param log_file: 要查询的数据库文件。
    :param table: 传感器所在的数据库表名。
    :param channel: 通道名称。
    :return: 表示该通道的 token。
    """
    q1 = f"""
        SELECT token
        FROM {table}
        WHERE channel == '{channel}';
    """
    row = execute_one(q1, (), log_file)

    if row is None:
        raise RuntimeError(f"未找到通道 {channel} 在表 {table} 中！")

    return str(row['token'].hex())


def get_images_from_lidar_tokens(
    log_file: str,
    tokens: List[str],
    channels: List[str],
    lookahead_window_us: int = 50000,
    lookback_window_us: int = 50000,
) -> Generator[Image, None, None]:
    """
    根据提供的 lidar_pc token 获取对应摄像头通道的图像数据。

    注意：lookahead_window_us 和 lookback_window_us 默认值为 50000 微秒（即 0.05 秒），
          总搜索窗口为 0.1 秒，围绕 lidar_pc 时间戳居中。
          因为 NuPlanDB 中 lidar_pc 是 20Hz，image 是 10Hz，所以需要这个窗口来保证能查到最近的图像。

    示例：
    iteration: 0    1    2    3   [4]   5    6
    timestamp: 0   0.05 0.1  0.15 0.2  0.25 0.3
    lidar_pc:  |    |    |    |    |    |    |
    Images:    |         |         |         |
    search window:           [---------]

    我们设置搜索窗口为 0.1 秒，以确保能检索到正确的图像数据。

    :param log_file: 要查询的日志文件。
    :param tokens: lidar_pc token 列表。
    :param channels: 要查询的摄像头通道列表。
    :param lookahead_window_us: [us] 向前查找时间窗口。
    :param lookback_window_us: [us] 向后查找时间窗口。
    :return: 图像数据 Image 实例的生成器。
    """
    query = f"""
            SELECT
                img.token,
                img.next_token,
                img.prev_token,
                img.ego_pose_token,
                img.camera_token,
                img.filename_jpg,
                img.timestamp,
                cam.channel
            FROM image AS img
              INNER JOIN lidar_pc AS lpc
                ON img.timestamp <= lpc.timestamp + ?
                AND img.timestamp >= lpc.timestamp - ?
              INNER JOIN camera AS cam
                ON cam.token = img.camera_token
            WHERE cam.channel IN ({('?,'*len(channels))[:-1]}) AND lpc.token IN ({('?,'*len(tokens))[:-1]})
            ORDER BY lpc.timestamp ASC;
    """
    args = [lookahead_window_us, lookback_window_us]
    args += channels  # type: ignore
    args += [bytearray.fromhex(t) for t in tokens]  # type: ignore

    for row in execute_many(query, args, log_file):
        yield Image.from_db_row(row)


def get_cameras(
    log_file: str,
    channels: List[str],
) -> Generator[Camera, None, None]:
    """
    获取指定通道的摄像头数据。
    
    :param log_file: 要查询的日志文件。
    :param channels: 要查询的通道列表。
    :return: Camera 实例的生成器。
    """
    query = f"""
            SELECT *
            FROM camera AS cam
            WHERE cam.channel IN ({('?,'*len(channels))[:-1]})
    """
    for row in execute_many(query, channels, log_file):
        yield Camera.from_db_row(row)
