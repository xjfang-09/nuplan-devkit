from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Type


@dataclass
class BoxParameters:
    """描述二维矩形框尺寸的基类（如车辆轮廓）"""

    width: float  # [m] 矩形宽度
    length: float  # [m] 矩形长度

    @property
    def half_width(self) -> float:
        """
        获取宽度的一半。
        :return: 宽度的一半值。
        """
        return self.width / 2.0

    @property
    def half_length(self) -> float:
        """
        获取长度的一半。
        :return: 长度的一半值。
        """
        return self.length / 2.0


class VehicleParameters(BoxParameters):
    """
    表示车辆参数的类。
    """

    def __init__(
        self,
        width: float,
        front_length: float,
        rear_length: float,
        cog_position_from_rear_axle: float,
        wheel_base: float,
        vehicle_name: str,
        vehicle_type: str,
        height: Optional[float] = None,
    ):
        """
        :param width: [m] 车辆包围盒的宽度。
        :param front_length: [m] 后轴到前保险杠的距离。
        :param rear_length: [m] 后轴到后保险杠的距离。
        :param cog_position_from_rear_axle: [m] 质心（COG）相对于后轴的位置。
        :param wheel_base: [m] 车辆轴距。
        :param vehicle_name: 车辆名称。
        :param vehicle_type: 车辆类型。
        :param height: [m] 车辆包围盒的高度。
        """
        self.width = width
        self.front_length = front_length  # [m] 后轴到前保险杠的距离
        self.rear_length = rear_length  # [m] 后轴到后保险杠的距离
        self.wheel_base = wheel_base
        self.length = front_length + rear_length  # 总长度
        self.cog_position_from_rear_axle = cog_position_from_rear_axle  # [m] COG 相对于后轴的位置
        self.height = height
        self.vehicle_name = vehicle_name
        self.vehicle_type = vehicle_type

    def __reduce__(self) -> Tuple[Type[VehicleParameters], Tuple[Any, ...]]:
        """
        返回用于序列化该类的构造函数及参数。
        :return: tuple 包含类和构造参数。
        """
        return self.__class__, (
            self.width,
            self.front_length,
            self.rear_length,
            self.cog_position_from_rear_axle,
            self.wheel_base,
            self.vehicle_name,
            self.vehicle_type,
            self.height,
        )

    @property
    def rear_axle_to_center(self) -> float:
        """
        获取后轴到车辆几何中心的距离。
        :return: [m] 距离值。
        """
        return self.half_length - self.rear_length

    @property
    def length_cog_to_front_axle(self) -> float:
        """
        获取质心到前轴的距离。
        :return: [m] 距离值。
        """
        return self.wheel_base - self.cog_position_from_rear_axle

    def __hash__(self) -> int:
        """
        :return: 对车辆参数计算 hash 值。
        """
        return hash(
            (
                self.vehicle_name,
                self.vehicle_type,
                self.width,
                self.front_length,
                self.rear_length,
                self.cog_position_from_rear_axle,
                self.wheel_base,
                self.height,
            )
        )

    def __str__(self) -> str:
        """
        :return: 本类对象的字符串表示。
        """
        return (
            f"VehicleParameters(vehicle_name={self.vehicle_name}, vehicle_type={self.vehicle_type}, "
            f"width={self.width}, front_length={self.front_length}, "
            f"rear_length={self.rear_length}, cog_position_from_rear_axle={self.cog_position_from_rear_axle}, "
            f"wheel_base={self.wheel_base}, height={self.height}, width={self.width})"
        )


def get_pacifica_parameters() -> VehicleParameters:
    """
    获取 Chrysler Pacifica 的车辆参数。
    :return: 包含 Pacifica 参数的 VehicleParameters 实例。
    """
    return VehicleParameters(
        vehicle_name="pacifica",
        vehicle_type="gen1",
        width=1.1485 * 2.0,
        front_length=4.049,
        rear_length=1.127,
        wheel_base=3.089,
        cog_position_from_rear_axle=1.67,
        height=1.777,
    )