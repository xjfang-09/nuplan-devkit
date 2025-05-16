from typing import List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


def _validate_waypoints(waypoints: List[InterpolatableState]) -> None:
    """
    确保用于插值的路径点是有效的
    如果路径点为空或它们不是单调递增的，则抛出异常
    :param waypoints: 要插值的路径点列表
    """
    if not waypoints:
        raise RuntimeError("路径点为空！")

    if not np.all(np.diff([w.time_us for w in waypoints]) > 0):
        raise ValueError(f"路径点不是单调递增的: {[w.time_us for w in waypoints]}！")


def _compute_desired_time_steps(
    start_timestamp: int, end_timestamp: int, horizon_len_s: float, interval_s: float
) -> Tuple[npt.NDArray[np.float64], int]:
    """
    计算所需的采样时间步
    :param start_timestamp: [微秒] 起始时间戳
    :param end_timestamp: [微秒] 结束时间戳
    :param horizon_len_s: [秒] 时间范围长度
    :param interval_s: [秒] 状态之间的间隔
    :return: 时间戳数组和所需的长度
    """
    # 提取所需的时间戳
    num_future_boxes = int(horizon_len_s / interval_s)
    num_target_timestamps = num_future_boxes + 1  # 包括当前帧 t0 的时间戳
    return np.linspace(start=start_timestamp, stop=end_timestamp, num=num_target_timestamps), num_target_timestamps


def _interpolate_waypoints(
    waypoints: List[InterpolatableState], target_timestamps: npt.NDArray[np.float64], pad_with_none: bool = True
) -> List[Optional[InterpolatableState]]:
    """
    根据所需的时间戳插值路径点
    :param waypoints: 要插值的路径点
    :param target_timestamps: 所需的采样时间戳
    :param pad_with_none: 如果为 True，则无法插值的状态将用 None 替代
    :return: 插值后的路径点列表，如果无法插值，则替换为 None
    """
    # 插值轨迹
    trajectory = InterpolatedTrajectory(waypoints)
    if pad_with_none:
        return [
            trajectory.get_state_at_time(TimePoint(t)) if trajectory.is_in_range(TimePoint(t)) else None
            for t in target_timestamps
        ]
    return [
        trajectory.get_state_at_time(TimePoint(t)) for t in target_timestamps if trajectory.is_in_range(TimePoint(t))
    ]


def interpolate_future_waypoints(
    waypoints: List[InterpolatableState], horizon_len_s: float, interval_s: float
) -> List[Optional[InterpolatableState]]:
    """
    插值未来的路径点。如果提供的路径点不足，则追加 None
    :param waypoints: 路径点列表，至少需要一个
    :param horizon_len_s: [秒] 到未来的时间距离
    :param interval_s: [秒] 两个状态之间的间隔
    :return: 插值后的路径点
    """
    _validate_waypoints(waypoints)

    # 提取所需的时间戳
    start_timestamp = waypoints[0].time_us
    end_timestamp = int(start_timestamp + horizon_len_s * 1e6)
    target_timestamps, num_future_boxes = _compute_desired_time_steps(
        start_timestamp, end_timestamp, horizon_len_s=horizon_len_s, interval_s=interval_s
    )

    if len(waypoints) == 1:
        # 如果轨迹太短，则不进行插值，仅追加 None
        return waypoints + cast(List[Optional[InterpolatableState]], [None] * (num_future_boxes - 1))

    # 插值轨迹
    return _interpolate_waypoints(waypoints, target_timestamps)


def interpolate_past_waypoints(
    waypoints: List[InterpolatableState], horizon_len_s: float, interval_s: float
) -> List[Optional[InterpolatableState]]:
    """
    插值过去的路径点。假设它们仍然是单调递增的。
    如果提供的路径点不足，则追加 None
    :param waypoints: 路径点列表，至少需要一个
    :param horizon_len_s: [秒] 到过去的时间距离
    :param interval_s: [秒] 两个状态之间的间隔
    :return: 插值后的路径点
    """
    _validate_waypoints(waypoints)

    # 提取所需的时间戳
    end_timestamp = waypoints[-1].time_us
    start_timestamp = max(int(end_timestamp - horizon_len_s * 1e6), 0)
    target_timestamps, num_future_boxes = _compute_desired_time_steps(
        start_timestamp, end_timestamp, horizon_len_s=horizon_len_s, interval_s=interval_s
    )

    if len(waypoints) == 1:
        # 如果轨迹太短，则不进行插值，仅追加 None
        return cast(List[Optional[InterpolatableState]], [None] * (num_future_boxes - 1)) + waypoints

    # 插值轨迹
    sampled_trajectory = _interpolate_waypoints(waypoints, target_timestamps)
    # 最后一个状态必须存在！
    if not sampled_trajectory[-1]:
        raise RuntimeError("轨迹的最后一个状态必须存在！")
    return sampled_trajectory