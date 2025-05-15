from __future__ import annotations

from functools import cached_property

from nuplan.common.actor_state.oriented_box import OrientedBox, OrientedBoxPointType
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.geometry.transform import translate_longitudinally


class CarFootprint(OrientedBox):
    """该类表示车辆的语义信息，包含几何形状和关键点信息。"""

    def __init__(self, center: StateSE2, vehicle_parameters: VehicleParameters):
        """
        :param center: 指定参考系下的自车姿态（位置和航向）
        :param vehicle_parameters: 自车参数对象
        """
        super().__init__(
            center=center,
            width=vehicle_parameters.width,
            length=vehicle_parameters.length,
            height=vehicle_parameters.height,
        )
        self._vehicle_parameters = vehicle_parameters

    @property
    def vehicle_parameters(self) -> VehicleParameters:
        """
        获取与该轮廓对应的车辆参数。
        :return: 车辆参数对象。
        """
        return self._vehicle_parameters

    def get_point_of_interest(self, point_of_interest: OrientedBoxPointType) -> Point2D:
        """
        获取指定的兴趣点坐标。
        :param point_of_interest: 查询的车辆兴趣点类型。
        :return: 对应兴趣点的二维坐标。
        """
        return self.corner(point_of_interest)

    @property
    def oriented_box(self) -> OrientedBox:
        """
        获取自车的 OrientedBox。
        :return: OrientedBox 类型的对象。
        """
        return self

    @property
    def rear_axle_to_center_dist(self) -> float:
        """
        获取后轴到质心的距离。
        :return: 后轴到 COG 的距离。
        """
        return float(self._vehicle_parameters.rear_axle_to_center)

    @cached_property
    def rear_axle(self) -> StateSE2:
        """
        获取后轴的位置姿态。
        :return: SE2 坐标下的后轴位置。
        """
        return translate_longitudinally(self.oriented_box.center, -self.rear_axle_to_center_dist)

    @classmethod
    def build_from_rear_axle(cls, rear_axle_pose: StateSE2, vehicle_parameters: VehicleParameters) -> CarFootprint:
        """
        从后轴构建车辆轮廓。
        :param rear_axle_pose: 后轴的姿态（StateSE2）。
        :param vehicle_parameters: 车辆参数。
        :return: 构建完成的 CarFootprint 实例。
        """
        center = translate_longitudinally(rear_axle_pose, vehicle_parameters.rear_axle_to_center)
        return cls(center=center, vehicle_parameters=vehicle_parameters)

    @classmethod
    def build_from_cog(cls, cog_pose: StateSE2, vehicle_parameters: VehicleParameters) -> CarFootprint:
        """
        从质心构建车辆轮廓。
        :param cog_pose: 质心的姿态（StateSE2）。
        :param vehicle_parameters: 车辆参数。
        :return: 构建完成的 CarFootprint 实例。
        """
        cog_to_center = vehicle_parameters.rear_axle_to_center - vehicle_parameters.cog_position_from_rear_axle
        center = translate_longitudinally(cog_pose, cog_to_center)
        return cls(center=center, vehicle_parameters=vehicle_parameters)

    @classmethod
    def build_from_center(cls, center: StateSE2, vehicle_parameters: VehicleParameters) -> CarFootprint:
        """
        从车辆几何中心构建车辆轮廓。
        :param center: 车辆几何中心的姿态。
        :param vehicle_parameters: 车辆参数。
        :return: 构建完成的 CarFootprint 实例。
        """
        return cls(center=center, vehicle_parameters=vehicle_parameters)