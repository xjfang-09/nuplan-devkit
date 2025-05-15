from __future__ import annotations

from functools import cached_property
from typing import Iterable, List, Union

from nuplan.common.actor_state.agent_state import AgentState
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState, get_acceleration_shifted, get_velocity_shifted
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.common.utils.split_state import SplitState


class EgoState(InterpolatableState):
    """表示自车（ego）当前状态及其动态属性。"""

    def __init__(
        self,
        car_footprint: CarFootprint,
        dynamic_car_state: DynamicCarState,
        tire_steering_angle: float,
        is_in_auto_mode: bool,
        time_point: TimePoint,
    ):
        """
        :param car_footprint: 自车的车辆轮廓（CarFootprint）
        :param dynamic_car_state: 自车的当前动态状态
        :param tire_steering_angle: 当前轮胎转向角度
        :param is_in_auto_mode: 表示该状态是否来自自动驾驶模式下的车辆
        :param time_point: 状态的时间戳
        """
        self._car_footprint = car_footprint
        self._tire_steering_angle = tire_steering_angle
        self._is_in_auto_mode = is_in_auto_mode
        self._time_point = time_point
        self._dynamic_car_state = dynamic_car_state

    @cached_property
    def waypoint(self) -> Waypoint:
        """
        :return: 与此 ego 状态对应的路径点（Waypoint）
        """
        return Waypoint(
            time_point=self.time_point,
            oriented_box=self.car_footprint,
            velocity=self.dynamic_car_state.rear_axle_velocity_2d,
        )

    @staticmethod
    def deserialize(vector: List[Union[int, float]], vehicle: VehicleParameters) -> EgoState:
        """
        反序列化对象，保持顺序以兼容旧版本格式。
        :param vector: 用于反序列化的变量列表。
        :param vehicle: 车辆参数。
        """
        if len(vector) != 9:
            raise RuntimeError(f'期望长度为9的向量，实际长度为 {len(vector)}')

        return EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(vector[1], vector[2], vector[3]),
            rear_axle_velocity_2d=StateVector2D(vector[4], vector[5]),
            rear_axle_acceleration_2d=StateVector2D(vector[6], vector[7]),
            tire_steering_angle=vector[8],
            time_point=TimePoint(int(vector[0])),
            vehicle_parameters=vehicle,
        )

    def __iter__(self) -> Iterable[Union[int, float]]:
        """返回 ego 的所有参数迭代器"""
        return iter(
            (
                self.time_us,
                self.rear_axle.x,
                self.rear_axle.y,
                self.rear_axle.heading,
                self.dynamic_car_state.rear_axle_velocity_2d.x,
                self.dynamic_car_state.rear_axle_velocity_2d.y,
                self.dynamic_car_state.rear_axle_acceleration_2d.x,
                self.dynamic_car_state.rear_axle_acceleration_2d.y,
                self.tire_steering_angle,
            )
        )

    def to_split_state(self) -> SplitState:
        """继承自父类，请参考 superclass 文档"""
        linear_states = [
            self.time_us,
            self.rear_axle.x,
            self.rear_axle.y,
            self.dynamic_car_state.rear_axle_velocity_2d.x,
            self.dynamic_car_state.rear_axle_velocity_2d.y,
            self.dynamic_car_state.rear_axle_acceleration_2d.x,
            self.dynamic_car_state.rear_axle_acceleration_2d.y,
            self.tire_steering_angle,
        ]
        angular_states = [self.rear_axle.heading]
        fixed_state = [self.car_footprint.vehicle_parameters]

        return SplitState(linear_states, angular_states, fixed_state)

    @staticmethod
    def from_split_state(split_state: SplitState) -> EgoState:
        """继承自父类，请参考 superclass 文档"""
        if len(split_state) != 10:
            raise RuntimeError(f'期望大小为10的状态向量，实际长度为 {len(split_state)}')

        return EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(
                split_state.linear_states[1], split_state.linear_states[2], split_state.angular_states[0]
            ),
            rear_axle_velocity_2d=StateVector2D(split_state.linear_states[3], split_state.linear_states[4]),
            rear_axle_acceleration_2d=StateVector2D(split_state.linear_states[5], split_state.linear_states[6]),
            tire_steering_angle=split_state.linear_states[7],
            time_point=TimePoint(int(split_state.linear_states[0])),
            vehicle_parameters=split_state.fixed_states[0],
        )

    @property
    def is_in_auto_mode(self) -> bool:
        """
        :return: 如果处于自动模式返回 True，否则返回 False。
        """
        return self._is_in_auto_mode

    @property
    def car_footprint(self) -> CarFootprint:
        """
        获取自车的车辆轮廓（CarFootprint）。
        :return: 自车的车辆轮廓。
        """
        return self._car_footprint

    @property
    def tire_steering_angle(self) -> float:
        """
        获取自车轮胎的转向角度。
        :return: 自车轮胎的转向角度。
        """
        return self._tire_steering_angle

    @property
    def center(self) -> StateSE2:
        """
        获取自车中心位置的姿态（质心）。
        :return: 自车中心姿态。
        """
        return self._car_footprint.oriented_box.center

    @property
    def rear_axle(self) -> StateSE2:
        """
        获取自车后轴位置的姿态（后轴中间）。
        :return: 自车后轴姿态。
        """
        return self.car_footprint.rear_axle

    @property
    def time_point(self) -> TimePoint:
        """
        获取 ego 状态的时间戳。
        :return: ego 状态的时间戳。
        """
        return self._time_point

    @property
    def time_us(self) -> int:
        """
        获取以微秒为单位的时间。
        :return: [us] 时间值。
        """
        return int(self.time_point.time_us)

    @property
    def time_seconds(self) -> float:
        """
        获取以秒为单位的时间。
        :return: [s] 时间值。
        """
        return float(self.time_us * 1e-6)

    @property
    def dynamic_car_state(self) -> DynamicCarState:
        """
        获取自车的动态状态。
        :return: 自车的动态状态。
        """
        return self._dynamic_car_state

    @property
    def scene_object_metadata(self) -> SceneObjectMetadata:
        """
        :return: 创建场景对象元数据
        """
        return SceneObjectMetadata(token='ego', track_token="ego", track_id=-1, timestamp_us=self.time_us)

    @cached_property
    def agent(self) -> AgentState:
        """
        将 EgoState 转换为 Agent 对象。
        :return: 包含 EgoState 参数的 Agent 对象。
        """
        return AgentState(
            metadata=self.scene_object_metadata,
            tracked_object_type=TrackedObjectType.EGO,
            oriented_box=self.car_footprint.oriented_box,
            velocity=self.dynamic_car_state.center_velocity_2d,
        )

    @classmethod
    def build_from_rear_axle(
        cls,
        rear_axle_pose: StateSE2,
        rear_axle_velocity_2d: StateVector2D,
        rear_axle_acceleration_2d: StateVector2D,
        tire_steering_angle: float,
        time_point: TimePoint,
        vehicle_parameters: VehicleParameters,
        is_in_auto_mode: bool = True,
        angular_vel: float = 0.0,
        angular_accel: float = 0.0,
        tire_steering_rate: float = 0.0,
    ) -> EgoState:
        """
        使用原始参数初始化，假设参考系为后轴。
        :param rear_axle_pose: 后轴的位置姿态
        :param rear_axle_velocity_2d: 后轴的速度矢量
        :param rear_axle_acceleration_2d: 后轴的加速度矢量
        :param angular_vel: 自车的角速度
        :param angular_accel: 自车的角加速度
        :param tire_steering_angle: 轮胎转向角度
        :param is_in_auto_mode: 是否处于自动模式，默认为 True
        :param time_point: ego 状态的时间戳
        :param vehicle_parameters: 车辆参数
        :param tire_steering_rate: 轮胎转向速率 [rad/s]
        :return: 初始化完成的 EgoState 实例
        """
        car_footprint = CarFootprint.build_from_rear_axle(
            rear_axle_pose=rear_axle_pose, vehicle_parameters=vehicle_parameters
        )
        dynamic_ego_state = DynamicCarState.build_from_rear_axle(
            rear_axle_to_center_dist=car_footprint.rear_axle_to_center_dist,
            rear_axle_velocity_2d=rear_axle_velocity_2d,
            rear_axle_acceleration_2d=rear_axle_acceleration_2d,
            angular_velocity=angular_vel,
            angular_acceleration=angular_accel,
            tire_steering_rate=tire_steering_rate,
        )

        return cls(
            car_footprint=car_footprint,
            dynamic_car_state=dynamic_ego_state,
            tire_steering_angle=tire_steering_angle,
            time_point=time_point,
            is_in_auto_mode=is_in_auto_mode,
        )

    @classmethod
    def build_from_center(
        cls,
        center: StateSE2,
        center_velocity_2d: StateVector2D,
        center_acceleration_2d: StateVector2D,
        tire_steering_angle: float,
        time_point: TimePoint,
        vehicle_parameters: VehicleParameters,
        is_in_auto_mode: bool = True,
        angular_vel: float = 0.0,
        angular_accel: float = 0.0,
    ) -> EgoState:
        """
        使用原始参数初始化自车状态，假设参考系为中心坐标系（center frame）
        :param center: 自车中心的姿态（pose）
        :param center_velocity_2d: 自车中心的速度矢量
        :param center_acceleration_2d: 自车中心的加速度矢量
        :param tire_steering_angle: 轮胎转向角度
        :param time_point: ego 状态的时间戳
        :param vehicle_parameters: 车辆参数
        :param is_in_auto_mode: 如果车辆处于自动驾驶模式则为 True，默认为 True
        :param angular_vel: 自车角速度，默认为 0.0
        :param angular_accel: 自车角加速度，默认为 0.0
        :return: 初始化完成的 EgoState 实例
        """
        car_footprint = CarFootprint.build_from_center(center, vehicle_parameters)
        rear_axle_to_center_dist = car_footprint.rear_axle_to_center_dist
        displacement = StateVector2D(-rear_axle_to_center_dist, 0.0)
        rear_axle_velocity_2d = get_velocity_shifted(displacement, center_velocity_2d, angular_vel)
        rear_axle_acceleration_2d = get_acceleration_shifted(
            displacement, center_acceleration_2d, angular_vel, angular_accel
        )

        dynamic_ego_state = DynamicCarState.build_from_rear_axle(
            rear_axle_to_center_dist=rear_axle_to_center_dist,
            rear_axle_velocity_2d=rear_axle_velocity_2d,
            rear_axle_acceleration_2d=rear_axle_acceleration_2d,
            angular_velocity=angular_vel,
            angular_acceleration=angular_accel,
        )

        return cls(
            car_footprint=car_footprint,
            dynamic_car_state=dynamic_ego_state,
            tire_steering_angle=tire_steering_angle,
            time_point=time_point,
            is_in_auto_mode=is_in_auto_mode,
        )


class EgoStateDot(EgoState):
    """
    表示 EgoState 的动力学状态。该类主要是为了提高代码可读性而存在。
    """

    pass