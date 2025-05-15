from __future__ import annotations

import math
from functools import cached_property
from typing import Tuple

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import StateVector2D


def get_velocity_shifted(
    displacement: StateVector2D, ref_velocity: StateVector2D, ref_angular_vel: float
) -> StateVector2D:
    """
    计算刚体上某点相对于参考点的速度。
    :param displacement: [m] 从参考点到查询点的位移向量。
    :param ref_velocity: [m/s] 参考点的速度向量。
    :param ref_angular_vel: [rad/s] 刚体绕垂直轴的角速度。
    :return: [m/s] 给定位移下的速度向量。
    """
    velocity_shift_term: npt.NDArray[np.float64] = np.array(
        [-displacement.y * ref_angular_vel, displacement.x * ref_angular_vel]
    )
    return StateVector2D(*(ref_velocity.array + velocity_shift_term))


def get_acceleration_shifted(
    displacement: StateVector2D, ref_accel: StateVector2D, ref_angular_vel: float, ref_angular_accel: float
) -> StateVector2D:
    """
    计算刚体上某点相对于参考点的加速度。
    :param displacement: [m] 从参考点到查询点的位移向量。
    :param ref_accel: [m/s^2] 参考点的加速度向量。
    :param ref_angular_vel: [rad/s] 刚体绕垂直轴的角速度。
    :param ref_angular_accel: [rad/s^2] 刚体绕垂直轴的角加速度。
    :return: [m/s^2] 给定位移下的加速度向量。
    """
    centripetal_acceleration_term = displacement.array * ref_angular_vel**2
    angular_acceleration_term = displacement.array * ref_angular_accel

    return StateVector2D(*(ref_accel.array + centripetal_acceleration_term + angular_acceleration_term))


def _get_beta(steering_angle: float, wheel_base: float) -> float:
    """
    计算 beta 值，即后轴到质心的角度（瞬时旋转中心）。
    :param steering_angle: [rad] 车辆转向角度。
    :param wheel_base: [m] 轴距。
    :return: [rad] beta 的值。
    """
    beta = math.atan2(math.tan(steering_angle), wheel_base)
    return beta


def _projected_velocities_from_cog(beta: float, cog_speed: float) -> Tuple[float, float]:
    """
    使用自行车运动学模型从 COG 推导出后轴的速度。
    :param beta: [rad] 后轴到 COG 的角度。
    :param cog_speed: [m/s] COG 处的速度大小。
    :return: 后轴处的纵向与横向速度 [m/s]。
    """
    # 根据模型假设，COG 纵向速度等于后轴纵向速度
    rear_axle_forward_velocity = math.cos(beta) * cog_speed  # [m/s]
    # 横向速度为 0
    rear_axle_lateral_velocity = 0

    return rear_axle_forward_velocity, rear_axle_lateral_velocity


def _angular_velocity_from_cog(
    cog_speed: float, length_rear_axle_to_cog: float, beta: float, steering_angle: float
) -> float:
    """
    使用自行车运动学模型计算自车角速度。
    :param cog_speed: [m/s] COG 处的速度大小。
    :param length_rear_axle_to_cog: [m] 后轴到 COG 的距离。
    :param beta: [rad] 后轴到 COG 的角度。
    :param steering_angle: [rad] 轮胎转向角度。
    """
    return (cog_speed / length_rear_axle_to_cog) * math.cos(beta) * math.tan(steering_angle)


def _project_accelerations_from_cog(
    rear_axle_longitudinal_velocity: float, angular_velocity: float, cog_acceleration: float, beta: float
) -> Tuple[float, float]:
    """
    使用自行车运动学模型从 COG 推导出后轴加速度。
    :param rear_axle_longitudinal_velocity: [m/s] COG 处的纵向速度。
    :param angular_velocity: [rad/s] COG 处的角速度。
    :param cog_acceleration: [m/s^2] COG 处的加速度大小。
    :param beta: [rad] 后轴到 COG 的角度。
    :return: 后轴处的纵向和横向加速度 [m/s^2]。
    """
    # 刚体假设下，可以从 COG 推导加速度
    rear_axle_longitudinal_acceleration = math.cos(beta) * cog_acceleration  # [m/s^2]

    # 向心加速度 a = v² / R，角速度 ω = v / R
    rear_axle_lateral_acceleration = rear_axle_longitudinal_velocity * angular_velocity  # [m/s^2]

    return rear_axle_longitudinal_acceleration, rear_axle_lateral_acceleration


class DynamicCarState:
    """包含自车各类动态属性的类。"""

    def __init__(
        self,
        rear_axle_to_center_dist: float,
        rear_axle_velocity_2d: StateVector2D,
        rear_axle_acceleration_2d: StateVector2D,
        angular_velocity: float = 0.0,
        angular_acceleration: float = 0.0,
        tire_steering_rate: float = 0.0,
    ):
        """
        :param rear_axle_to_center_dist: [m] 自车后轴到几何中心的距离（正数）。
        :param rear_axle_velocity_2d: [m/s] 后轴处的速度矢量。
        :param rear_axle_acceleration_2d: [m/s^2] 后轴处的加速度矢量。
        :param angular_velocity: [rad/s] 自车的角速度。
        :param angular_acceleration: [rad/s^2] 自车的角加速度。
        :param tire_steering_rate: [rad/s] 轮胎转向速率。
        """
        self._rear_axle_to_center_dist = rear_axle_to_center_dist
        self._angular_velocity = angular_velocity
        self._angular_acceleration = angular_acceleration
        self._rear_axle_velocity_2d = rear_axle_velocity_2d
        self._rear_axle_acceleration_2d = rear_axle_acceleration_2d
        self._tire_steering_rate = tire_steering_rate

    @property
    def rear_axle_velocity_2d(self) -> StateVector2D:
        """
        获取后轴处的速度矢量。
        :return: StateVector2D 类型的速度矢量。
        """
        return self._rear_axle_velocity_2d

    @property
    def rear_axle_acceleration_2d(self) -> StateVector2D:
        """
        获取后轴处的加速度矢量。
        :return: StateVector2D 类型的加速度矢量。
        """
        return self._rear_axle_acceleration_2d

    @cached_property
    def center_velocity_2d(self) -> StateVector2D:
        """
        获取自车几何中心处的速度矢量。
        :return: StateVector2D 类型的几何中心速度矢量。
        """
        displacement = StateVector2D(self._rear_axle_to_center_dist, 0.0)
        return get_velocity_shifted(displacement, self.rear_axle_velocity_2d, self.angular_velocity)

    @cached_property
    def center_acceleration_2d(self) -> StateVector2D:
        """
        获取自车几何中心处的加速度矢量。
        :return: StateVector2D 类型的几何中心加速度矢量。
        """
        displacement = StateVector2D(self._rear_axle_to_center_dist, 0.0)
        return get_acceleration_shifted(
            displacement, self.rear_axle_acceleration_2d, self.angular_velocity, self.angular_acceleration
        )

    @property
    def angular_velocity(self) -> float:
        """
        获取自车的角速度。
        :return: [rad/s] 角速度值。
        """
        return self._angular_velocity

    @property
    def angular_acceleration(self) -> float:
        """
        获取自车的角加速度。
        :return: [rad/s^2] 角加速度值。
        """
        return self._angular_acceleration

    @property
    def tire_steering_rate(self) -> float:
        """
        获取轮胎的转向速率。
        :return: [rad/s] 转向速率。
        """
        return self._tire_steering_rate

    @cached_property
    def speed(self) -> float:
        """
        获取自车几何中心的速度大小。
        :return: [m/s] 一维速度值。
        """
        return float(self._rear_axle_velocity_2d.magnitude())

    @cached_property
    def acceleration(self) -> float:
        """
        获取自车几何中心的加速度大小。
        :return: [m/s^2] 一维加速度值。
        """
        return float(self._rear_axle_acceleration_2d.magnitude())

    def __eq__(self, other: object) -> bool:
        """
        比较两个 DynamicCarState 实例是否数值相近。
        :param other: 待比较的对象。
        :return: 如果对象几乎相等则返回 True。
        """
        if not isinstance(other, DynamicCarState):
            return NotImplemented

        return (
            self.rear_axle_velocity_2d == other.rear_axle_velocity_2d
            and self.rear_axle_acceleration_2d == other.rear_axle_acceleration_2d
            and math.isclose(self._angular_acceleration, other._angular_acceleration)
            and math.isclose(self._angular_velocity, other._angular_velocity)
            and math.isclose(self._rear_axle_to_center_dist, other._rear_axle_to_center_dist)
            and math.isclose(self._tire_steering_rate, other._tire_steering_rate)
        )

    def __repr__(self) -> str:
        """返回该类的字符串表示，用于调试"""
        return (
            f"Rear Axle| velocity: {self.rear_axle_velocity_2d}, acceleration: {self.rear_axle_acceleration_2d}\n"
            f"Center   | velocity: {self.center_velocity_2d}, acceleration: {self.center_acceleration_2d}\n"
            f"angular velocity: {self.angular_velocity}, angular acceleration: {self._angular_acceleration}\n"
            f"rear_axle_to_center_dist: {self._rear_axle_to_center_dist} \n"
            f"_tire_steering_rate: {self._tire_steering_rate} \n"
        )

    @staticmethod
    def build_from_rear_axle(
        rear_axle_to_center_dist: float,
        rear_axle_velocity_2d: StateVector2D,
        rear_axle_acceleration_2d: StateVector2D,
        angular_velocity: float = 0.0,
        angular_acceleration: float = 0.0,
        tire_steering_rate: float = 0.0,
    ) -> DynamicCarState:
        """
        使用后轴参数构建自车状态。
        :param rear_axle_to_center_dist: [m] 几何中心到后轴的距离。
        :param rear_axle_velocity_2d: [m/s] 后轴处的速度矢量。
        :param rear_axle_acceleration_2d: [m/s^2] 后轴处的加速度矢量。
        :param angular_velocity: [rad/s] 角速度。
        :param angular_acceleration: [rad/s^2] 角加速度。
        :param tire_steering_rate: [rad/s] 轮胎转向速率。
        :return: 构建完成的 DynamicCarState 实例。
        """
        return DynamicCarState(
            rear_axle_to_center_dist=rear_axle_to_center_dist,
            rear_axle_velocity_2d=rear_axle_velocity_2d,
            rear_axle_acceleration_2d=rear_axle_acceleration_2d,
            angular_velocity=angular_velocity,
            angular_acceleration=angular_acceleration,
            tire_steering_rate=tire_steering_rate,
        )

    @staticmethod
    def build_from_cog(
        wheel_base: float,
        rear_axle_to_center_dist: float,
        cog_speed: float,
        cog_acceleration: float,
        steering_angle: float,
        angular_acceleration: float = 0.0,
        tire_steering_rate: float = 0.0,
    ) -> DynamicCarState:
        """
        使用 COG 参数构建自车状态。
        :param wheel_base: [m] 轴距。
        :param rear_axle_to_center_dist: [m] 几何中心到后轴的距离。
        :param cog_speed: [m/s] COG 处的速度大小。
        :param cog_acceleration: [m/s^2] COG 处的加速度大小。
        :param steering_angle: [rad] 轮胎转向角度。
        :param angular_acceleration: [rad/s^2] 角加速度。
        :param tire_steering_rate: [rad/s] 轮胎转向速率。
        :return: 构建完成的 DynamicCarState 实例。
        """
        # 在运动学模型假设下：推导所需其他状态
        beta = _get_beta(steering_angle, wheel_base)

        rear_axle_longitudinal_velocity, rear_axle_lateral_velocity = _projected_velocities_from_cog(beta, cog_speed)

        angular_velocity = _angular_velocity_from_cog(cog_speed, wheel_base, beta, steering_angle)

        # 根据运动学模型推导后轴加速度
        longitudinal_acceleration, lateral_acceleration = _project_accelerations_from_cog(
            rear_axle_longitudinal_velocity, angular_velocity, cog_acceleration, beta
        )

        return DynamicCarState(
            rear_axle_to_center_dist=rear_axle_to_center_dist,
            rear_axle_velocity_2d=StateVector2D(rear_axle_longitudinal_velocity, rear_axle_lateral_velocity),
            rear_axle_acceleration_2d=StateVector2D(longitudinal_acceleration, lateral_acceleration),
            angular_velocity=angular_velocity,
            angular_acceleration=angular_acceleration,
            tire_steering_rate=tire_steering_rate,
        )