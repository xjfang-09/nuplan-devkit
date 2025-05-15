from __future__ import annotations

import abc
from typing import List, Optional, Tuple

from shapely.geometry import LineString, Point, Polygon

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.maps_datatypes import IntersectionType, LaneConnectorType, StopLineType


class AbstractMapObject(abc.ABC):
    """
    所有地图对象的基本接口定义。
    """

    def __init__(self, object_id: str):
        """
        构造函数。
        :param object_id: 地图对象的唯一标识符。
        """
        self.id = str(object_id)


class PolygonMapObject(AbstractMapObject):
    """
    用于表示可以用多边形表示的地图对象的基类。
    """

    @property
    @abc.abstractmethod
    def polygon(self) -> Polygon:
        """
        返回地图对象的多边形表示。
        :return: 表示地图对象的 Polygon 对象。
        """
        pass

    def contains_point(self, point: Point2D) -> bool:
        """
        判断指定点是否在该地图对象的多边形范围内。
        :return: 如果点位于多边形内返回 True，否则返回 False。
        """
        return bool(self.polygon.contains(Point(point.x, point.y)))


class GraphEdgeMapObject(PolygonMapObject):
    """
    表示地图图结构中边类型的地图对象。
    """

    @property
    @abc.abstractmethod
    def incoming_edges(self) -> List[GraphEdgeMapObject]:
        """
        获取连接到此边的入边。
        :return: GraphEdgeMapObject 的列表。
        """
        pass

    @property
    @abc.abstractmethod
    def outgoing_edges(self) -> List[GraphEdgeMapObject]:
        """
        获取从此边出发的出边。
        :return: GraphEdgeMapObject 的列表。
        """
        pass

    @property
    @abc.abstractmethod
    def parallel_edges(self) -> List[GraphEdgeMapObject]:
        """
        获取与此边平行的边（包括自己）。
        :return: GraphEdgeMapObject 的列表。
        """
        pass


class LaneGraphEdgeMapObject(GraphEdgeMapObject):
    """
    可以作为地图图结构一部分，并且内部包含基准线的地图对象。
    """

    @property
    @abc.abstractmethod
    def incoming_edges(self) -> List[LaneGraphEdgeMapObject]:  # type: ignore
        """
        获取连接到此车道边的入边。
        :return: LaneGraphEdgeMapObject 的列表。
        """
        pass

    @property
    @abc.abstractmethod
    def outgoing_edges(self) -> List[LaneGraphEdgeMapObject]:  # type: ignore
        """
        获取从此车道边出发的出边。
        :return: LaneGraphEdgeMapObject 的列表。
        """
        pass

    @property
    @abc.abstractmethod
    def baseline_path(self) -> PolylineMapObject:
        """
        获取车道的中心线路径。
        :return: 车道的中心线。
        """
        pass

    @property
    @abc.abstractmethod
    def left_boundary(self) -> PolylineMapObject:
        """
        获取车道的左边界。
        :return: 车道左侧边界。
        """
        pass

    @property
    @abc.abstractmethod
    def right_boundary(self) -> PolylineMapObject:
        """
        获取车道的右边界。
        :return: 车道右侧边界。
        """
        pass

    @property
    @abc.abstractmethod
    def speed_limit_mps(self) -> Optional[float]:
        """
        获取车道的速度限制。
        :return: [m/s] 速度限制值，如果没有定义则返回 None。
        """
        pass

    @abc.abstractmethod
    def get_roadblock_id(self) -> str:
        """
        获取包含该车道的道路块 ID。
        :return: 包含车道的道路块 ID。
        """
        pass

    @abc.abstractmethod
    def parent(self) -> RoadBlockGraphEdgeMapObject:
        """
        获取包含该车道的父级道路块对象。
        :return: RoadBlockGraphEdgeMapObject 类型的对象。
        """
        pass

    @abc.abstractmethod
    def has_traffic_lights(self) -> bool:
        """
        判断该边是否受交通信号灯控制。
        :return: 如果受控制返回 True，否则返回 False。
        """
        pass

    @property
    @abc.abstractmethod
    def stop_lines(self) -> List['StopLine']:
        """
        获取与该车道关联的所有停止线。
        :return: StopLine 的列表。
        """
        pass

    def is_same_roadblock(self, other: 'Lane') -> bool:
        """
        判断另一个车道是否和当前车道处于同一个道路块。
        :param other: 待比较的车道。
        :return: 如果处于同一道路块返回 True。
        """
        return self.get_roadblock_id() == other.get_roadblock_id()

    def is_adjacent_to(self, other: 'Lane') -> bool:
        """
        判断另一个车道是否和当前车道相邻。
        :param other: 待比较的车道。
        :return: 如果相邻返回 True。
        """
        return self.is_same_roadblock(other) and (
            self.right_boundary.id == other.left_boundary.id or self.left_boundary.id == other.right_boundary.id
        )

    @abc.abstractmethod
    def is_left_of(self, other: 'Lane') -> bool:
        """
        判断当前车道是否在指定车道的左侧。
        :param other: 被比较的车道。
        :return: 如果是左侧返回 True。
        :raise AssertionError: 如果车道不在同一道路块中抛出异常。
        """
        pass

    @abc.abstractmethod
    def is_right_of(self, other: 'Lane') -> bool:
        """
        判断当前车道是否在指定车道的右侧。
        :param other: 被比较的车道。
        :return: 如果是右侧返回 True。
        :raise AssertionError: 如果车道不在同一道路块中抛出异常。
        """
        pass

    @property
    @abc.abstractmethod
    def adjacent_edges(self) -> Tuple[Optional['LaneGraphEdgeMapObject'], Optional['LaneGraphEdgeMapObject']]:
        """
        获取相邻的车道边。
        :return: 元组 (左侧车道, 右侧车道)。
        """
        pass

    @abc.abstractmethod
    def get_width_left_right(self, point: Point2D, include_outside: bool = False) -> Tuple[float, float]:
        """
        获取从给定点到车道左右边界的距离。
        :param point: 点坐标。
        :param include_outside: 是否允许点在车道外。
        :return: 左右两侧的距离。
        """
        pass

    @abc.abstractmethod
    def oriented_distance(self, point: Point2D) -> float:
        """
        计算点相对于边的定向距离。
        :param point: 点坐标。
        :return: 距离值。正数代表左侧，负数代表右侧。
        """
        pass


class Lane(LaneGraphEdgeMapObject):
    """
    表示车道的类。
    """

    def __init__(self, lane_id: str):
        """
        构造函数。
        :param lane_id: 车道的唯一标识符。
        """
        super().__init__(lane_id)

    def has_traffic_lights(self) -> bool:
        """继承自父类方法。"""
        return False

    @property
    def stop_lines(self) -> List['StopLine']:
        """继承自父类方法。"""
        return []

    @abc.abstractmethod
    def index(self) -> int:
        """
        获取车道在其父级道路块中的索引位置（从1开始计数）。
        :return: 车道索引。
        """
        pass


class LaneConnector(LaneGraphEdgeMapObject):
    """
    表示车道连接器的类。
    """

    def __init__(self, lane_connector_id: str):
        """
        构造函数。
        :param lane_connector_id: 车道连接器的唯一标识符。
        """
        super().__init__(lane_connector_id)

    @property
    def adjacent_edges(self) -> Tuple[Optional['LaneGraphEdgeMapObject'], Optional['LaneGraphEdgeMapObject']]:
        """继承自父类方法。"""
        return None, None

    @property
    @abc.abstractmethod
    def turn_type(self) -> LaneConnectorType:
        """
        获取车道连接器的转向类型。
        :return: 转向类型或 None。
        """
        pass


class PolylineMapObject(AbstractMapObject):
    """
    表示可由折线描述的地图对象。
    """

    def __init__(self, path_id: str):
        """
        构造函数。
        :param path_id: 折线的唯一标识符。
        """
        super().__init__(path_id)

    @property
    @abc.abstractmethod
    def linestring(self) -> LineString:
        """
        获取折线的 Linestring 表示。
        :return: Linestring 对象。
        """
        pass

    @property
    @abc.abstractmethod
    def length(self) -> float:
        """
        获取折线的长度。
        :return: [m] 长度值。
        """
        pass

    @property
    @abc.abstractmethod
    def discrete_path(self) -> List[StateSE2]:
        """
        获取折线的离散化路径表示。
        :return: StateSE2 对象的列表。
        """
        pass

    @abc.abstractmethod
    def get_nearest_arc_length_from_position(self, point: Point2D) -> float:
        """
        获取离给定位置最近的弧长。
        :param point: [m] 点坐标。
        :return: [m] 弧长值。
        """
        pass

    @abc.abstractmethod
    def get_nearest_pose_from_position(self, point: Point2D) -> StateSE2:
        """
        获取离给定位置最近的位姿。
        :param point: [m] 点坐标。
        :return: 最近的位姿。
        """
        pass

    @abc.abstractmethod
    def get_curvature_at_arc_length(self, arc_length: float) -> float:
        """
        获取折线上某弧长处的曲率。
        :param arc_length: [m] 弧长值。
        :return: [1/m] 曲率值。
        """
        pass

    def get_nearest_curvature_from_position(self, point: Point2D) -> float:
        """
        获取离给定位置最近的曲率值。
        :param point: [m] 点坐标。
        :return: [1/m] 曲率值。
        """
        return self.get_curvature_at_arc_length(self.get_nearest_arc_length_from_position(point))


class RoadBlockGraphEdgeMapObject(GraphEdgeMapObject):
    """
    表示可以作为地图图结构的一部分、并且包含多个 LaneGraphEdgeMapObject 的地图对象。
    """

    @property
    @abc.abstractmethod
    def incoming_edges(self) -> List['RoadBlockGraphEdgeMapObject']:  # type: ignore
        """
        获取连接到此边的入边。
        :return: RoadBlockGraphEdgeMapObject 的列表。
        """
        pass

    @property
    @abc.abstractmethod
    def outgoing_edges(self) -> List['RoadBlockGraphEdgeMapObject']:  # type: ignore
        """
        获取从此边出发的出边。
        :return: RoadBlockGraphEdgeMapObject 的列表。
        """
        pass

    @property
    @abc.abstractmethod
    def interior_edges(self) -> List[LaneGraphEdgeMapObject]:
        """
        获取组成该容器边的边缘。
        :return: LaneGraphEdgeMapObject 的列表。
        """
        pass

    @property
    @abc.abstractmethod
    def children_stop_lines(self) -> List['StopLine']:
        """
        获取该道路块内的所有停止线。
        :return: StopLine 的列表。
        """
        pass

    def intersection(self) -> Optional['Intersection']:
        """
        获取该道路块对应的交叉路口（如果存在）。
        :return: Intersection 对象或 None。
        """
        pass


class StopLine(PolygonMapObject):
    """
    表示停止线的类。
    """

    def __init__(self, stop_line_id: str, stop_line_type: StopLineType) -> None:
        """
        构造函数。
        :param stop_line_id: 停止线的唯一标识符。
        :param stop_line_type: 停止线子类型，如 PED_CROSSING、STOP_SIGN、TRAFFIC_LIGHT 等。
        """
        super().__init__(stop_line_id)
        self.stop_line_type = stop_line_type

    @property
    @abc.abstractmethod
    def intersection_from(self) -> 'Intersection':
        """
        获取相关的交叉路口。
        :return: 关联的 Intersection 对象。
        """
        pass

    @property
    @abc.abstractmethod
    def layer_type(self) -> StopLineType:
        """
        获取停止线的类型。
        :return: StopLineType 类型。
        """
        pass

    @property
    @abc.abstractmethod
    def parent(self) -> RoadBlockGraphEdgeMapObject:
        """
        获取包含该停止线的父级 RoadBlockGraphEdgeMapObject。
        :return: 父级 RoadBlockGraphEdgeMapObject。
        """
        pass


class Intersection(PolygonMapObject):
    """
    表示交叉路口的类。
    """

    def __init__(self, intersection_id: str, intersection_type: IntersectionType) -> None:
        """
        构造函数。
        :param intersection_id: 交叉路口的唯一标识符。
        :param intersection_type: 交叉路口类型，如 DEFAULT、TRAFFIC_LIGHT、STOP_SIGN 等。
        """
        super().__init__(intersection_id)
        self.intersection_type = intersection_type

    @property
    @abc.abstractmethod
    def interior_edges(self) -> List[RoadBlockGraphEdgeMapObject]:
        """
        获取交叉路口内包含的 RoadBlockGraphEdgeMapObject。
        :return: RoadBlockGraphEdgeMapObject 列表。
        """
        pass

    @property
    @abc.abstractmethod
    def incoming_edges(self) -> List[Lane]:
        """
        获取连接到该交叉路口的车道。
        :return: Lane 对象列表。
        """
        pass

    @property
    @abc.abstractmethod
    def is_signaled(self) -> bool:
        """
        判断交叉路口是否有信号控制。
        :return: 如果是信号控制路口返回 True，否则返回 False。
        """
        pass