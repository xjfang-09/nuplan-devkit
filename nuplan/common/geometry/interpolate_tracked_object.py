from typing import List, Union, cast

from nuplan.common.actor_state.agent_temporal_state import AgentTemporalState
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.geometry.interpolate_state import interpolate_future_waypoints, interpolate_past_waypoints
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory


def interpolate_agent(agent: AgentTemporalState, horizon_len_s: float, interval_s: float) -> AgentTemporalState:
    """
    根据预定义的长度和间隔插值代理的未来预测和过去轨迹
    :param agent: 要插值的代理
    :param horizon_len_s: [秒] 预测的时间范围
    :param interval_s: [秒] 两个状态之间的间隔
    :return: 插值后的代理，其中缺失的路径点用 None 替代
    """
    interpolated_agent = agent
    if interpolated_agent.predictions:
        interpolated_agent.predictions = [
            PredictedTrajectory(
                waypoints=interpolate_future_waypoints(
                    mode.waypoints, horizon_len_s=horizon_len_s, interval_s=interval_s
                ),
                probability=mode.probability,
            )
            for mode in interpolated_agent.predictions
        ]

    past_trajectory = interpolated_agent.past_trajectory
    if past_trajectory:
        interpolated_agent.past_trajectory = PredictedTrajectory(
            waypoints=interpolate_past_waypoints(
                past_trajectory.waypoints, horizon_len_s=horizon_len_s, interval_s=interval_s
            ),
            probability=past_trajectory.probability,
        )
    return interpolated_agent


def interpolate_tracks(
    tracked_objects: Union[TrackedObjects, List[TrackedObject]], horizon_len_s: float, interval_s: float
) -> List[TrackedObject]:
    """
    插值代理的预测和过去轨迹，如果状态不足，则添加 None！
    :param tracked_objects: 要插值的代理
    :param horizon_len_s: [秒] 从初始路径点开始的时间范围
    :param interval_s: [秒] 两个状态之间的间隔
    :return: 插值后的代理
    """
    all_tracked_objects = (
        tracked_objects if isinstance(tracked_objects, TrackedObjects) else TrackedObjects(tracked_objects)
    )
    return [
        interpolate_agent(agent, horizon_len_s=horizon_len_s, interval_s=interval_s)
        for agent in all_tracked_objects.get_agents()
    ] + cast(List[TrackedObject], all_tracked_objects.get_static_objects())