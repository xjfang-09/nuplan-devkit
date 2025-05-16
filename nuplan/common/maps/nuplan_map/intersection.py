from functools import cached_property
from typing import List, Tuple

from shapely.geometry import Polygon

from nuplan.common.maps.abstract_map_objects import Intersection, Lane, LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import IntersectionType, VectorLayer
from nuplan.common.maps.nuplan_map.utils import get_row_with_value


class NuPlanIntersection(Intersection):
    """
    NuPlanMap 中对交叉路口（Intersection）的实现。
    """

    def __init__(self, intersection_id: str, intersections_df: VectorLayer) -> None:
        """
        初始化交叉路口对象。
        :param intersection_id: 交叉路口的唯一标识符。
        :param intersections_df: 包含地图中所有交叉路口的 geopandas GeoDataframe。
        """
        self._intersections_df = intersections_df
        self._intersection = get_row_with_value(self._intersections_df, "fid", intersection_id)
        super().__init__(intersection_id, IntersectionType.DEFAULT)  # 当前 GPKG 不支持交叉路口类型

    @cached_property
    def polygon(self) -> Polygon:
        """
        获取交叉路口的几何多边形表示。
        :return: Shapely Polygon 实例。
        """
        return self._intersection.geometry

    @cached_property
    def interior_edges(self) -> List[LaneGraphEdgeMapObject]:
        """
        获取交叉路口内部的车道图边（LaneGraphEdgeMapObject）列表。
        :return: 内部车道图边对象列表。
        """
        raise NotImplementedError

    @cached_property
    def incoming_edges(self) -> List[Lane]:
        """
        获取进入该交叉路口的车道列表。
        :return: 进入车道列表。
        """
        raise NotImplementedError

    @cached_property
    def center(self) -> Tuple[float, float]:
        """
        获取交叉路口中心坐标。
        :return: (x, y) 形式的中心点坐标元组。
        """
        raise NotImplementedError

    @cached_property
    def is_signaled(self) -> bool:
        """
        判断该交叉路口是否是信号灯控制路口。
        :return: 如果是信号灯路口返回 True，否则返回 False。
        """
        raise NotImplementedError