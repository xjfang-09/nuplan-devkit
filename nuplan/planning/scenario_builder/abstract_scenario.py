from __future__ import annotations

import abc
from typing import Generator, List, Optional, Set

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimeDuration, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatuses, Transform
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, SensorChannel, Sensors
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class AbstractScenario(abc.ABC):
    """
    所有数据库中通用场景的接口定义。
    """

    @property
    @abc.abstractmethod
    def token(self) -> str:
        """
        获取该场景的唯一标识符。
        :return: 表示唯一 token 的字符串。
        """
        pass

    @property
    @abc.abstractmethod
    def log_name(self) -> str:
        """
        获取创建此场景的日志名称。
        :return: 日志名称字符串。
        """
        pass

    @property
    @abc.abstractmethod
    def scenario_name(self) -> str:
        """
        获取该场景的名称，例如 extraction_xxxx。
        :return: 场景名称字符串。
        """
        pass

    @property
    @abc.abstractmethod
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """
        查询自车的车辆参数。
        :return: 包含车辆参数的 VehicleParameters 对象。
        """
        pass

    @property
    @abc.abstractmethod
    def scenario_type(self) -> str:
        """
        获取场景类型。
        :return: 场景类型字符串，如 lane_change、lane_follow 等。
        """
        pass

    @property
    @abc.abstractmethod
    def map_api(self) -> AbstractMap:
        """
        返回该场景使用的地图 API。
        :return: AbstractMap 实例。
        """
        pass

    @property
    @abc.abstractmethod
    def database_interval(self) -> float:
        """
        数据库的时间间隔（秒）。
        :return: [s] 时间间隔。
        """
        pass

    @abc.abstractmethod
    def get_number_of_iterations(self) -> int:
        """
        获取该场景包含的帧数。
        :return: 整数，表示场景中的帧数量。
        """
        pass

    @abc.abstractmethod
    def get_time_point(self, iteration: int) -> TimePoint:
        """
        获取指定迭代次数的时间戳。
        :param iteration: 场景中的迭代次数，0 <= iteration < number_of_iterations。
        :return: 全局时间点。
        """
        pass

    @property
    def start_time(self) -> TimePoint:
        """
        获取场景开始时间。
        :return: 开始时间点。
        """
        return self.get_time_point(0)

    @property
    def end_time(self) -> TimePoint:
        """
        获取场景结束时间。
        :return: 结束时间点。
        """
        return self.get_time_point(self.get_number_of_iterations() - 1)

    @property
    def duration_s(self) -> TimeDuration:
        """
        获取场景持续时间（秒）。
        :return: 场景起始和结束时间之间的差值（秒）。
        """
        return TimeDuration.from_s(self.end_time.time_s - self.start_time.time_s)

    @abc.abstractmethod
    def get_lidar_to_ego_transform(self) -> Transform:
        """
        获取激光雷达与自车间的变换矩阵。
        :return: [4x4] 旋转和平移矩阵。
        """
        pass

    @abc.abstractmethod
    def get_mission_goal(self) -> Optional[StateSE2]:
        """
        获取远期目标状态（通常在场景终点后超过100米以上）。
        :return: 最终状态的 StateSE2。
        """
        pass

    @abc.abstractmethod
    def get_route_roadblock_ids(self) -> List[str]:
        """
        获取组成目标路线的 RoadBlock ID 列表。
        :return: 道路块 ID 字符串列表。
        """
        pass

    @abc.abstractmethod
    def get_expert_goal_state(self) -> StateSE2:
        """
        获取专家驾驶员在场景结束时达到的目标状态。
        :return: 最终状态的 StateSE2。
        """
        pass

    @abc.abstractmethod
    def get_tracked_objects_at_iteration(
        self,
        iteration: int,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """
        获取指定迭代次数下的追踪对象。
        :param iteration: 场景内的迭代次数。
        :param future_trajectory_sampling: 如果需要未来轨迹采样参数。
        :return: 检测追踪结果 DetectionsTracks。
        """
        pass

    @abc.abstractmethod
    def get_tracked_objects_within_time_window_at_iteration(
        self,
        iteration: int,
        past_time_horizon: float,
        future_time_horizon: float,
        filter_track_tokens: Optional[Set[str]] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """
        获取从当前迭代向前一段时间内和向后一段时间内的所有追踪对象。
        可选地根据 track_token 过滤结果。结果按对象类型、时间戳、track_token 排序。

        :param iteration: 要查询的场景迭代次数。
        :param past_time_horizon: [s] 向前查找的时间范围。
        :param future_time_horizon: [s] 向后查找的时间范围。
        :param filter_track_tokens: 若提供，则只返回匹配这些 token 的对象。
        :param future_trajectory_sampling: agent未来真实轨迹采样参数。
        :return: 检索到的检测追踪数据。
        """
        pass

    @property
    def initial_tracked_objects(self) -> DetectionsTracks:
        """
        获取初始时刻的追踪对象。
        :return: DetectionsTracks 对象。
        """
        return self.get_tracked_objects_at_iteration(0)

    @abc.abstractmethod
    def get_sensors_at_iteration(self, iteration: int, channels: Optional[List[SensorChannel]] = None) -> Sensors:
        """
        获取指定迭代次数的传感器数据。
        :param iteration: 场景中的迭代次数。
        :param channels: 要返回的传感器通道。
        :return: Sensors 对象。
        """
        pass

    @property
    def initial_sensors(self) -> Sensors:
        """
        获取初始时刻的传感器数据（如点云）。
        :return: Sensors 对象。
        """
        return self.get_sensors_at_iteration(0)

    @abc.abstractmethod
    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """
        获取指定迭代次数下专家驾驶的自车状态。
        :param iteration: 场景中的迭代次数。
        :return: 自车状态 EgoState。
        """
        pass

    @property
    def initial_ego_state(self) -> EgoState:
        """
        获取初始时刻的自车状态。
        :return: 自车状态 EgoState。
        """
        return self.get_ego_state_at_iteration(0)

    @abc.abstractmethod
    def get_traffic_light_status_at_iteration(self, iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        """
        获取指定迭代次数下的交通灯状态。
        :param iteration: 场景中的迭代次数。
        :return: 当前迭代的交通灯状态。
        """
        pass

    @abc.abstractmethod
    def get_past_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        """
        获取过去时间段内的交通灯状态历史。
        
        :param iteration: 场景中的迭代次数。
        :param time_horizon: [s] 查找过去状态的时间范围。
        :param num_samples: 历史记录条目数，若为 None 则从数据库推断。
        :return: 过去交通灯状态的生成器。
        """
        pass

    @abc.abstractmethod
    def get_future_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        """
        获取未来时间段内的交通灯状态历史。
        
        :param iteration: 场景中的迭代次数。
        :param time_horizon: [s] 查找未来状态的时间范围。
        :param num_samples: 历史记录条目数，若为 None 则从数据库推断。
        :return: 未来交通灯状态的生成器。
        """
        pass

    def get_expert_ego_trajectory(self) -> Generator[EgoState, None, None]:
        """
        获取专家驾驶员的历史轨迹。
        :return: 自车状态序列。
        """
        return (self.get_ego_state_at_iteration(index) for index in range(self.get_number_of_iterations()))

    def get_ego_trajectory_slice(self, start_idx: int, end_idx: int) -> Generator[EgoState, None, None]:
        """
        获取专家驾驶员在 start_idx 和 end_idx 之间的轨迹。
        :param start_idx: 轨迹起始索引。
        :param end_idx: 轨迹结束索引。
        :return: 自车状态序列。
        """
        return (self.get_ego_state_at_iteration(index) for index in range(start_idx, end_idx))

    @abc.abstractmethod
    def get_future_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        """
        获取未来时间戳。
        :param iteration: 场景中的迭代次数。
        :param time_horizon: [s] 查找未来状态的时间范围。
        :param num_samples: 要获取的条目数。
        :return: 与数据库最接近匹配的未来时间戳。
        """
        pass

    @abc.abstractmethod
    def get_past_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        """
        获取过去时间戳。
        :param iteration: 场景中的迭代次数。
        :param time_horizon: [s] 查找过去状态的时间范围。
        :param num_samples: 要获取的条目数。
        :return: 与数据库最接近匹配的过去时间戳。
        """
        pass

    @abc.abstractmethod
    def get_ego_future_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        """
        获取自车未来的轨迹。
        :param iteration: 场景中的迭代次数。
        :param time_horizon: [s] 查找未来轨迹的时间范围。
        :param num_samples: 要获取的条目数。
        :return: 与数据库最接近匹配的自车未来轨迹。
        """
        pass

    @abc.abstractmethod
    def get_ego_past_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        """
        获取自车过去的轨迹。
        :param iteration: 场景中的迭代次数。
        :param time_horizon: [s] 查找过去轨迹的时间范围。
        :param num_samples: 要获取的条目数。
        :return: 与数据库最接近匹配的自车过去轨迹。
        """
        pass

    @abc.abstractmethod
    def get_past_sensors(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        channels: Optional[List[SensorChannel]] = None,
    ) -> Generator[Sensors, None, None]:
        """
        获取过去时间段内的传感器数据。
        :param iteration: 场景中的迭代次数。
        :param time_horizon: [s] 查找过去数据的时间范围。
        :param num_samples: 要获取的数据条目数。
        :param channels: 要返回的传感器通道。
        :return: 与数据库最接近匹配的过去传感器数据。
        """
        pass

    @abc.abstractmethod
    def get_past_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        """
        获取过去时间段内的检测对象。
        :param iteration: 场景中的迭代次数。
        :param time_horizon: [s] 查找过去数据的时间范围。
        :param num_samples: 要获取的数据条目数。
        :param future_trajectory_sampling: agent未来轨迹采样参数。
        :return: 与数据库最接近匹配的过去检测对象。
        """
        pass

    @abc.abstractmethod
    def get_future_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        """
        获取未来时间段内的检测对象。
        :param iteration: 场景中的迭代次数。
        :param time_horizon: [s] 查找未来数据的时间范围。
        :param num_samples: 要获取的数据条目数。
        :param future_trajectory_sampling: agent未来轨迹采样参数。
        :return: 与数据库最接近匹配的未来检测对象。
        """
        pass