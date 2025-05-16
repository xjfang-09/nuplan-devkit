from __future__ import annotations

from typing import List, Optional

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory


class AgentTemporalState:
    """
    带有当前、多模态未来轨迹以及过去轨迹的参与体。
        未来轨迹的概率之和必须为 1.0。
        过去轨迹仅为单模态，模式概率为 1.0。
        过去轨迹的最后一个航点必须与当前位置一致（这里只检查时间戳）。
    """

    def __init__(
        self,
        initial_time_stamp: TimePoint,
        predictions: Optional[List[PredictedTrajectory]] = None,
        past_trajectory: Optional[PredictedTrajectory] = None,
    ):
        """
        初始化具有过去和未来轨迹的参与体时序状态
        :param initial_time_stamp: 当前检测的时间戳
        :param predictions: 未来多模态轨迹
        :param past_trajectory: 已经过的过去轨迹
        """
        self._initial_time_stamp = initial_time_stamp
        self.predictions: List[PredictedTrajectory] = predictions if predictions is not None else []
        self.past_trajectory = past_trajectory

    @property
    def previous_state(self) -> Optional[Waypoint]:
        """
        :return: 如果 agent 没有上一个状态则返回 None，否则返回上一个状态
        """
        # 至少需要 2 个状态，因为最后一个状态与当前状态相同
        if not self.past_trajectory or len(self.past_trajectory.valid_waypoints) < 2:
            return None
        return self.past_trajectory.waypoints[-2]

    @property
    def predictions(self) -> List[PredictedTrajectory]:
        """
        获取 agent 的预测轨迹
        :return: 轨迹列表
        """
        return self._predictions

    @predictions.setter
    def predictions(self, predicted_trajectories: List[PredictedTrajectory]) -> None:
        """
        设置预测轨迹，检查概率之和是否为 1。
        :param predicted_trajectories: 预测轨迹列表
        """
        if not predicted_trajectories:
            self._predictions = predicted_trajectories
            return
        # 健全性检查：如果提供了预测轨迹，概率之和必须为 1
        probability_sum = sum(prediction.probability for prediction in predicted_trajectories)
        if not abs(probability_sum - 1) < 1e-6 and predicted_trajectories:
            raise ValueError(f"所提供的轨迹概率之和不是 1，而是 {probability_sum:.2f}！")
        self._predictions = predicted_trajectories

    @property
    def past_trajectory(self) -> Optional[PredictedTrajectory]:
        """
        获取 agent 的过去轨迹
        :return: 轨迹
        """
        return self._past_trajectory

    @past_trajectory.setter
    def past_trajectory(self, past_trajectory: Optional[PredictedTrajectory]) -> None:
        """
        设置过去轨迹，检查概率之和是否为 1。
        :param past_trajectory: 已行驶的轨迹
        """
        if not past_trajectory:
            # 如果为 None，无需检查
            self._past_trajectory = past_trajectory
            return

        # 确保当前状态已设置！
        last_waypoint = past_trajectory.waypoints[-1]
        if not last_waypoint:
            raise RuntimeError("最后一个航点表示当前 agent 的状态，不应为 None！")

        # 健全性检查：最后一个航点的时间戳应与当前一致
        if last_waypoint.time_point != self._initial_time_stamp:
            raise ValueError(
                "所提供的轨迹未以当前 agent 状态结束！"
                f" {last_waypoint.time_us} != {self._initial_time_stamp}"
            )
        self._past_trajectory = past_trajectory