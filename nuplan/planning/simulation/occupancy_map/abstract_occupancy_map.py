from __future__ import annotations

import abc
from typing import List, Tuple, Union

from shapely.geometry import LineString, Polygon

Geometry = Union[Polygon, LineString]


class OccupancyMap(abc.ABC):
    """
    一个用于处理几何体之间空间关系的类。主要功能包括：
    1. 碰撞检测
    2. 查询最近的几何体
    """

    @abc.abstractmethod
    def get_nearest_entry_to(self, geometry_id: str) -> Tuple[str, Geometry, float]:
        """
        返回与查询几何体最近的几何体所在的行
        :param geometry_id: 查询几何体的 ID
        :return: 最近的几何体、对应的 ID 和到最近几何体的距离
        @raises AssertionError 如果占用图中不包含 geometry_id
        """
        pass

    @abc.abstractmethod
    def intersects(self, geometry: Geometry) -> OccupancyMap:
        """
        返回一个新的占用图，其中包含所有与给定几何体相交的几何体
        :param geometry: 要检查相交的几何体
        """
        pass

    @abc.abstractmethod
    def insert(self, geometry_id: str, geometry: Geometry) -> None:
        """
        向占用图中插入一个几何体
        :param geometry_id: 几何体的 ID
        :param geometry: 要插入的几何体
        """
        pass

    @abc.abstractmethod
    def get(self, geometry_id: str) -> Geometry:
        """
        获取与对应 geometry_id 关联的几何体
        :param geometry_id: 几何体的 ID
        """
        pass

    @abc.abstractmethod
    def set(self, geometry_id: str, geometry: Geometry) -> None:
        """
        使用新的几何体设置特定几何体
        :param geometry_id: 几何体的 ID
        :param geometry: 要设置的新几何体
        """
        pass

    @abc.abstractmethod
    def get_all_ids(self) -> List[str]:
        """
        返回占用图中所有几何体的 ID
        :return: 包含所有 ID 的字符串列表
        """

    @abc.abstractmethod
    def get_all_geometries(self) -> List[Geometry]:
        """
        返回占用图中所有的几何体
        :return: 包含所有几何体的 Geometry 列表
        """

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """
        :return: 占用图中的条目数量
        """
        pass

    @abc.abstractmethod
    def is_empty(self) -> bool:
        """
        :return: 如果占用图为空则返回 True
        """
        pass

    @abc.abstractmethod
    def contains(self, geometry_id: str) -> bool:
        """
        :return: 如果占用图中存在具有给定 ID 的几何体则返回 True
        """
        pass

    @abc.abstractmethod
    def remove(self, geometry_ids: List[str]) -> None:
        """
        移除与对应 geometry_ids 关联的几何体
        :param geometry_ids: 几何体的 ID 列表
        """
        pass

    def __len__(self) -> int:
        """支持 len()，返回地图中的条目数量。"""
        return self.size