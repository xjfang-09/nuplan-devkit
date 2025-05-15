from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List

import geopandas as gpd
import numpy as np
import numpy.typing as npt

Transform = npt.NDArray[np.float32]  # 4x4 齐次变换矩阵

PointCloud = npt.NDArray[np.float32]  # Nx4 的激光雷达点数组（TODO: 包装成 dataclass）

VectorLayer = gpd.GeoDataFrame


class SemanticMapLayer(IntEnum):
    """
    语义地图图层类型的枚举。
    """

    LANE = 0  # 车道
    INTERSECTION = 1  # 交叉路口
    STOP_LINE = 2  # 停止线
    TURN_STOP = 3  # 转弯停止线
    CROSSWALK = 4  # 人行横道
    DRIVABLE_AREA = 5  # 可行驶区域
    YIELD = 6  # 让行标志
    TRAFFIC_LIGHT = 7  # 交通信号灯
    STOP_SIGN = 8  # 停车标志
    EXTENDED_PUDO = 9  # 扩展上下客/货区域
    SPEED_BUMP = 10  # 减速带
    LANE_CONNECTOR = 11  # 车道连接器
    BASELINE_PATHS = 12  # 基准路径
    BOUNDARIES = 13  # 边界
    WALKWAYS = 14  # 步行道
    CARPARK_AREA = 15  # 停车场区域
    PUDO = 16  # 上下客/货区域
    ROADBLOCK = 17  # 道路块
    ROADBLOCK_CONNECTOR = 18  # 道路块连接器

    @classmethod
    def deserialize(cls, layer: str) -> SemanticMapLayer:
        """
        从字符串加载时反序列化类型。
        :param layer: 字符串形式的枚举名。
        :return: 对应的 SemanticMapLayer 枚举值。
        """
        return SemanticMapLayer.__members__[layer]


class LaneConnectorType(IntEnum):
    """
    车道连接器类型的枚举。
    """

    STRAIGHT = 0  # 直行
    LEFT = 1  # 左转
    RIGHT = 2  # 右转
    UTURN = 3  # U型转弯
    UNKNOWN = 4  # 未知


class StopLineType(IntEnum):
    """
    停止线类型的枚举。
    """

    PED_CROSSING = 0  # 人行横道
    STOP_SIGN = 1  # 停车标志
    TRAFFIC_LIGHT = 2  # 交通信号灯
    TURN_STOP = 3  # 转弯停止
    YIELD = 4  # 让行
    UNKNOWN = 5  # 未知


class PudoType(IntEnum):
    """
    上下客/货区域类型的枚举。
    """

    PICK_UP_DROP_OFF = 0  # 可上可下
    PICK_UP_ONLY = 1  # 仅可上车
    DROP_OFF_ONLY = 2  # 仅可下车
    UNKNOWN = 3  # 未知


class IntersectionType(IntEnum):
    """
    交叉路口类型的枚举。
    """

    DEFAULT = 0  # 默认
    TRAFFIC_LIGHT = 1  # 有交通信号灯
    STOP_SIGN = 2  # 有停车标志
    LANE_BRANCH = 3  # 车道分支
    LANE_MERGE = 4  # 车道合并
    PASS_THROUGH = 5  # 直接通过


class TrafficLightStatusType(IntEnum):
    """
    交通信号灯状态类型的枚举。
    """

    GREEN = 0  # 绿灯
    YELLOW = 1  # 黄灯
    RED = 2  # 红灯
    UNKNOWN = 3  # 未知

    def serialize(self) -> str:
        """
        序列化类型，用于保存。
        :return: 类型名称字符串。
        """
        return self.name

    @classmethod
    def deserialize(cls, key: str) -> TrafficLightStatusType:
        """
        从字符串加载时反序列化类型。
        :param key: 字符串形式的枚举名。
        :return: 对应的 TrafficLightStatusType 枚举值。
        """
        return TrafficLightStatusType.__members__[key]


@dataclass
class RasterLayer:
    """
    栅格地图图层的数据类封装。
    """

    data: npt.NDArray[np.uint8]  # 栅格图像数据（numpy 数组）
    precision: np.float64  # [m] 地图精度
    transform: Transform  # 从物理坐标到像素坐标的变换矩阵（4x4）


@dataclass
class VectorMap:
    """
    将 SemanticMapLayers 映射到对应 VectorLayer 的数据类。
    """

    layers: Dict[SemanticMapLayer, VectorLayer]  # type: ignore


@dataclass
class RasterMap:
    """
    将 SemanticMapLayers 映射到对应 RasterLayer 的数据类。
    """

    layers: Dict[SemanticMapLayer, RasterLayer]


@dataclass
class TrafficLightStatusData:
    """表示交通信号灯状态的数据类。"""

    status: TrafficLightStatusType  # 状态：绿灯、红灯等
    lane_connector_id: int  # 此交通信号灯所属车道连接器的 ID
    timestamp: int  # 时间戳

    def serialize(self) -> Dict[str, Any]:
        """
        序列化交通信号灯状态。
        :return: 字典格式的状态数据。
        """
        return {
            'status': self.status.serialize(),
            'lane_connector_id': self.lane_connector_id,
            'timestamp': self.timestamp,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> TrafficLightStatusData:
        """
        将字典数据反序列化为该类实例。
        :param data: 包含状态数据的字典。
        :return: TrafficLightStatusData 实例。
        """
        return TrafficLightStatusData(
            status=TrafficLightStatusType.deserialize(data['status']),
            lane_connector_id=data['lane_connector_id'],
            timestamp=data['timestamp'],
        )


@dataclass
class TrafficLightStatuses:
    """
    某个时间步下的所有交通信号灯状态集合。
    """

    traffic_lights: List[TrafficLightStatusData]