from __future__ import annotations

from functools import lru_cache
from typing import Any, Tuple, Type

from nuplan.common.maps.abstract_map_factory import AbstractMapFactory
from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
from nuplan.database.maps_db.imapsdb import IMapsDB


class NuPlanMapFactory(AbstractMapFactory):
    """
    基于 IMapsDB 接口创建地图的工厂类。
    """

    def __init__(self, maps_db: IMapsDB):
        """
        :param maps_db: 实现 IMapsDB 的实例，例如 GPKGMapsDB。
        """
        self._maps_db = maps_db

    def __reduce__(self) -> Tuple[Type[NuPlanMapFactory], Tuple[Any, ...]]:
        """
        用于对象序列化重建（如 pickle）时的提示信息。
        :return: 对象类型及构造参数。
        """
        return self.__class__, (self._maps_db,)

    def build_map_from_name(self, map_name: str) -> NuPlanMap:
        """
        根据地图名称构建地图接口。
        示例名称：'sg-one-north', 'us-ma-boston', 'us-nv-las-vegas-strip', 'us-pa-pittsburgh-hazelwood'
        :param map_name: 地图名称。
        :return: 构建完成的地图接口实例。
        """
        return NuPlanMap(self._maps_db, map_name.replace(".gpkg", ""))


@lru_cache(maxsize=2)
def get_maps_db(map_root: str, map_version: str) -> GPKGMapsDB:
    """
    从磁盘加载一个 MapsDB 实例。
    :param map_root: 地图数据根目录。
    :param map_version: 要加载的地图版本。
    :return: 加载完成的 MapsDB 对象。
    """
    return GPKGMapsDB(map_root=map_root, map_version=map_version)


@lru_cache(maxsize=32)
def get_maps_api(map_root: str, map_version: str, map_name: str) -> NuPlanMap:
    """
    获取与指定参数对应的地图 API 实例。
    :param map_root: 地图数据根目录。
    :param map_version: 要加载的地图版本。
    :param map_name: 要加载的地图名称。
    :return: 加载完成的 NuPlanMap 实例。
    """
    maps_db = get_maps_db(map_root, map_version)
    return NuPlanMap(maps_db, map_name.replace(".gpkg", ""))