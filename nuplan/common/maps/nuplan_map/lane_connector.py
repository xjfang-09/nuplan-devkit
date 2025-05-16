from functools import cached_property
from typing import List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

import nuplan.common.maps.nuplan_map.lane as lane
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import (
    LaneConnector,
    LaneGraphEdgeMapObject,
    PolylineMapObject,
    RoadBlockGraphEdgeMapObject,
    StopLine,
)
from nuplan.common.maps.maps_datatypes import LaneConnectorType, SemanticMapLayer, VectorLayer
from nuplan.common.maps.nuplan_map.polyline_map_object import NuPlanPolylineMapObject
from nuplan.common.maps.nuplan_map.stop_line import NuPlanStopLine
from nuplan.common.maps.nuplan_map.utils import get_row_with_value


class NuPlanLaneConnector(LaneConnector):
    """
    NuPlanMap 中 LaneConnector 的实现。
    """

    def __init__(
        self,
        lane_connector_id: str,
        lanes_df: VectorLayer,
        lane_connectors_df: VectorLayer,
        baseline_paths_df: VectorLayer,
        boundaries_df: VectorLayer,
        stop_lines_df: VectorLayer,
        lane_connector_polygon_df: VectorLayer,
        map_data: AbstractMap,
    ):
        """
        NuPlanLaneConnector 的构造函数。
        :param lane_connector_id: 车道连接器的唯一标识符。
        :param lanes_df: 包含地图中所有车道的 GeoDataFrame。
        :param lane_connectors_df: 包含地图中所有车道连接器的 GeoDataFrame。
        :param baseline_paths_df: 包含地图中所有基线的 GeoDataFrame。
        :param boundaries_df: 包含地图中所有边界的 GeoDataFrame。
        :param stop_lines_df: 包含地图中所有停止线的 GeoDataFrame。
        :param lane_connector_polygon_df: 包含车道连接器多边形的 GeoDataFrame。
        """
        super().__init__(lane_connector_id)
        self._lanes_df = lanes_df
        self._lane_connectors_df = lane_connectors_df
        self._baseline_paths_df = baseline_paths_df
        self._boundaries_df = boundaries_df
        self._stop_lines_df = stop_lines_df
        self._lane_connector_polygon_df = lane_connector_polygon_df
        self._lane_connector = None
        self._map_data = map_data

    @cached_property
    def incoming_edges(self) -> List[LaneGraphEdgeMapObject]:
        """从父类继承。"""
        incoming_lane_id = self._get_lane_connector()["exit_lane_fid"]

        return [
            lane.NuPlanLane(
                str(incoming_lane_id),
                self._lanes_df,
                self._lane_connectors_df,
                self._baseline_paths_df,
                self._boundaries_df,
                self._stop_lines_df,
                self._lane_connector_polygon_df,
                self._map_data,
            )
        ]

    @cached_property
    def outgoing_edges(self) -> List[LaneGraphEdgeMapObject]:
        """从父类继承。"""
        outgoing_lane_id = self._get_lane_connector()["entry_lane_fid"]

        return [
            lane.NuPlanLane(
                str(outgoing_lane_id),
                self._lanes_df,
                self._lane_connectors_df,
                self._baseline_paths_df,
                self._boundaries_df,
                self._stop_lines_df,
                self._lane_connector_polygon_df,
                self._map_data,
            )
        ]

    @cached_property
    def parallel_edges(self) -> List[LaneGraphEdgeMapObject]:
        """从父类继承。"""
        raise NotImplementedError

    @cached_property
    def baseline_path(self) -> PolylineMapObject:
        """从父类继承。"""
        return NuPlanPolylineMapObject(get_row_with_value(self._baseline_paths_df, "lane_connector_fid", self.id))

    @cached_property
    def left_boundary(self) -> PolylineMapObject:
        """从父类继承。"""
        boundary_fid = get_row_with_value(self._lane_connector_polygon_df, "lane_connector_fid", self.id)[
            "left_boundary_fid"
        ]
        return NuPlanPolylineMapObject(get_row_with_value(self._boundaries_df, "fid", str(boundary_fid)))

    @cached_property
    def right_boundary(self) -> PolylineMapObject:
        """从父类继承。"""
        boundary_fid = get_row_with_value(self._lane_connector_polygon_df, "lane_connector_fid", self.id)[
            "right_boundary_fid"
        ]
        return NuPlanPolylineMapObject(get_row_with_value(self._boundaries_df, "fid", str(boundary_fid)))

    @cached_property
    def speed_limit_mps(self) -> Optional[float]:
        """从父类继承。"""
        speed_limit = self._get_lane_connector()["speed_limit_mps"]
        is_valid = speed_limit == speed_limit and speed_limit is not None
        return float(speed_limit) if is_valid else None

    @cached_property
    def polygon(self) -> Polygon:
        """从父类继承。注意，多边形是从基线推断的。"""
        lane_connector_polygon_row = get_row_with_value(self._lane_connector_polygon_df, "lane_connector_fid", self.id)
        return lane_connector_polygon_row.geometry

    def is_left_of(self, other: LaneConnector) -> bool:
        """从父类继承。"""
        # 由于缺乏车道连接器的邻接信息，此方法始终返回 False
        return False

    def is_right_of(self, other: LaneConnector) -> bool:
        """从父类继承。"""
        # 由于缺乏车道连接器的邻接信息，此方法始终返回 False
        return False

    def get_roadblock_id(self) -> str:
        """从父类继承。"""
        return str(self._get_lane_connector()["lane_group_connector_fid"])

    @cached_property
    def parent(self) -> RoadBlockGraphEdgeMapObject:
        """从父类继承。"""
        return self._map_data.get_map_object(self.get_roadblock_id(), SemanticMapLayer.ROADBLOCK_CONNECTOR)

    def has_traffic_lights(self) -> bool:
        """从父类继承。"""
        return bool(self._get_lane_connector()["traffic_light_stop_line_fids"])

    @cached_property
    def stop_lines(self) -> List[StopLine]:
        """从父类继承。"""
        stop_line_ids = self._get_lane_connector()["traffic_light_stop_line_fids"]
        stop_line_ids = cast(List[str], stop_line_ids.replace(" ", "").split(","))

        candidate_stop_lines = [NuPlanStopLine(id_, self._stop_lines_df) for id_ in stop_line_ids if id_]

        # 此车道连接器没有关联的停止线
        if not candidate_stop_lines:
            return []

        stop_lines = [
            stop_line
            for stop_line in candidate_stop_lines
            if stop_line.polygon.intersects(self.baseline_path.linestring)
        ]

        # 如果交集检查成功，则返回停止线。
        if stop_lines:
            return stop_lines

        # 停止线未与车道连接器的基线相交。改为执行距离检查。
        def distance_to_stop_line(stop_line: StopLine) -> float:
            """
            计算车道连接器基线路径的第一个点与停止线之间的距离。
            :param stop_line: 要计算距离的停止线。
            :return: [m] 车道连接器第一个点与停止线多边形之间的距离。
            """
            start = Point(self.baseline_path.linestring.coords[0])
            return float(start.distance(stop_line.polygon))

        distances = [distance_to_stop_line(stop_line) for stop_line in candidate_stop_lines]

        return [candidate_stop_lines[np.argmin(distances)]]

    def turn_type(self) -> LaneConnectorType:
        """从父类继承。"""
        raise NotImplementedError

    def get_width_left_right(
        self, point: Point2D, include_outside: bool = False
    ) -> Tuple[Optional[float], Optional[float]]:
        """从父类继承。"""
        raise NotImplementedError

    def oriented_distance(self, point: Point2D) -> float:
        """从父类继承。"""
        raise NotImplementedError

    def _get_lane_connector(self) -> pd.Series:
        """
        从车道数据框中获取包含车道 ID 的系列。
        :return: 车道数据框中相应的系列。
        """
        if self._lane_connector is None:
            self._lane_connector = get_row_with_value(self._lane_connectors_df, "fid", self.id)

        return self._lane_connector