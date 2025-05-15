from __future__ import annotations

from functools import cached_property
from typing import List, Optional

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.agent_temporal_state import AgentTemporalState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory


class EgoTemporalState(AgentTemporalState):
    """
    包含自车当前状态及其过去和未来轨迹的时间状态。
    """

    def __init__(
        self,
        current_state: EgoState,
        past_trajectory: Optional[PredictedTrajectory] = None,
        predictions: Optional[List[PredictedTrajectory]] = None,
    ):
        """
        初始化时间状态。
        :param current_state: 自车当前状态。
        :param past_trajectory: 之前的轨迹，最后路径点应与当前状态位置相同。
        :param predictions: 多模态预测列表，或未来轨迹。
        """
        super().__init__(
            initial_time_stamp=current_state.time_point, predictions=predictions, past_trajectory=past_trajectory
        )
        self._ego_current_state = current_state

    @property
    def ego_current_state(self) -> EgoState:
        """
        获取自车当前状态。
        :return: EgoState 实例。
        """
        return self._ego_current_state

    @property
    def ego_previous_state(self) -> Optional[EgoState]:
        """
        如果存在，获取自车前一时刻的状态。
        :return: 返回类型正确的前一状态。
        """
        return self.previous_state

    @cached_property
    def agent(self) -> Agent:
        """
        将 EgoTemporalState 转换为 Agent 对象。
        :return: 包含 EgoState 参数的 Agent 实例。
        """
        return Agent(
            metadata=self.ego_current_state.scene_object_metadata,
            tracked_object_type=TrackedObjectType.EGO,
            oriented_box=self.ego_current_state.car_footprint.oriented_box,
            velocity=self.ego_current_state.dynamic_car_state.center_velocity_2d,
            past_trajectory=self.past_trajectory,
            predictions=self.predictions,
        )