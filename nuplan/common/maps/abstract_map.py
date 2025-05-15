from __future__ import annotations

import abc
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map_objects import (
    Intersection,
    Lane,
    LaneConnector,
    PolygonMapObject,
    RoadBlockGraphEdgeMapObject,
    StopLine,
)
from nuplan.common.maps.maps_datatypes import RasterLayer, RasterMap, SemanticMapLayer

MapObject = Union[Lane, LaneConnector, RoadBlockGraphEdgeMapObject, PolygonMapObject, Intersection, StopLine]


class AbstractMap(abc.ABC):
    """
    通用场景地图 API 的接口定义。
    """

    @abc.abstractmethod
    def get_available_map_objects(self) -> List[SemanticMapLayer]:
        """
        获取可用的地图对象类型。
        :return: SemanticMapLayer 类型的列表。
        """
        pass

    @abc.abstractmethod
    def get_available_raster_layers(self) -> List[SemanticMapLayer]:
        """
        获取可用的栅格图层类型。
        :return: SemanticMapLayer 类型的列表。
        """
        pass

    @abc.abstractmethod
    def get_raster_map_layer(self, layer: SemanticMapLayer) -> RasterLayer:
        """
        获取指定语义层的栅格图层。
        :param layer: 要查询的语义层。
        :return: RasterLayer 对象。返回一个映射到 RasterLayer 的字典。
        """
        pass

    @abc.abstractmethod
    def get_raster_map(self, layers: List[SemanticMapLayer]) -> RasterMap:
        """
        获取多个指定语义层的栅格图层。
        :param layers: 要查询的语义层列表。
        :return: RasterMap 对象。返回一个映射到 RasterLayer 的字典。
        """
        pass

    @property
    @abc.abstractmethod
    def map_name(self) -> str:
        """
        :return: 地图所在位置的名称。
        """
        pass

    @abc.abstractmethod
    def get_all_map_objects(self, point: Point2D, layer: SemanticMapLayer) -> List[MapObject]:
        """
        返回包含给定点 (x, y) 的语义层上的所有地图对象。
        :param point: [m] 全局坐标系下的 x、y 坐标。
        :param layer: 要查询的语义层。
        :return: 地图对象列表。
        """
        pass

    @abc.abstractmethod
    def get_one_map_object(self, point: Point2D, layer: SemanticMapLayer) -> Optional[MapObject]:
        """
        返回包含给定点 (x, y) 的语义层上的一个地图对象。
        :param point: [m] 全局坐标系下的 x、y 坐标。
        :param layer: 要查询的语义层。
        :return: 如果存在则返回一个地图对象，否则返回 None。
        @raise AssertionError 如果找到多个对象
        """
        pass

    @abc.abstractmethod
    def is_in_layer(self, point: Point2D, layer: SemanticMapLayer) -> bool:
        """
        检查给定点 (x, y) 是否在指定语义层内。
        :param point: [m] 全局坐标系下的 x、y 坐标。
        :param layer: 要查询的语义层。
        :return: 如果点在该层内返回 True，否则返回 False。
        @raise ValueError 如果语义层不存在
        """
        pass

    @abc.abstractmethod
    def get_proximal_map_objects(
        self, point: Point2D, radius: float, layers: List[SemanticMapLayer]
    ) -> Dict[SemanticMapLayer, List[MapObject]]:
        """
        提取指定点半径范围内的地图对象。
        :param point: [m] 全局坐标系下的 x、y 坐标。
        :param radius: [m] 向量地图查询半径。
        :param layers: 要检查的图层列表。
        :return: 映射语义图层到地图对象列表的字典。
        """
        pass

    @abc.abstractmethod
    def get_map_object(self, object_id: str, layer: SemanticMapLayer) -> Optional[MapObject]:
        """
        获取具有指定 ID 的地图对象。
        :param object_id: 要提取的地图对象唯一 ID。
        :param layer: 要查询的语义层。
        :return: 如果存在对应 ID 的地图对象则返回该对象，否则返回 None。
        """
        pass

    @abc.abstractmethod
    def get_distance_to_nearest_map_object(
        self, point: Point2D, layer: SemanticMapLayer
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        获取到最近目标表面的距离（米），距离是该点到表面上最近点的 L1 范数。
        :param point: [m] 全局坐标系下的 x、y 坐标。
        :param layer: 要查询的语义层。
        :return: 表面 ID 和距离值。如果不存在对应表面，则返回 -1 和 np.NaN。
        """
        pass

    @abc.abstractmethod
    def get_distance_to_nearest_raster_layer(self, point: Point2D, layer: SemanticMapLayer) -> float:
        """
        获取到最近栅格图层的距离（米），距离是该点到表面上最近点的 L1 范数。
        :param point: [m] 全局坐标系下的 x、y 坐标。
        :param layer: 要查询的语义层。
        :return: 如果存在返回距离值，否则返回 None。
        @raise ValueError 如果语义层不存在
        """
        pass

    @abc.abstractmethod
    def get_distances_matrix_to_nearest_map_object(
        self, points: List[Point2D], layer: SemanticMapLayer
    ) -> Optional[npt.NDArray[np.float64]]:
        """
        返回点列表与最近目标表面之间的距离矩阵（单位：米）。
        距离是该点到表面上最近点的 L1 范数。
        :param points: [m] 全局坐标系下的点列表。
        :param layer: 要查询的语义层。
        :return: 每个点到最近目标表面的最短距离数组。
        """
        pass

    @abc.abstractmethod
    def initialize_all_layers(self) -> None:
        """
        加载所有图层到向量地图中。
        """
        pass