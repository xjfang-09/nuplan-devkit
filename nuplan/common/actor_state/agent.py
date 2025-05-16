from __future__ import annotations

from typing import List, Optional

from nuplan.common.actor_state.agent_state import AgentState
from nuplan.common.actor_state.agent_temporal_state import AgentTemporalState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory


class Agent(AgentTemporalState, AgentState):
    """
    带有未来和过去轨迹的 AgentState。
    """

    def __init__(
        self,
        tracked_object_type: TrackedObjectType,
        oriented_box: OrientedBox,
        velocity: StateVector2D,
        metadata: SceneObjectMetadata,
        angular_velocity: Optional[float] = None,
        predictions: Optional[List[PredictedTrajectory]] = None,
        past_trajectory: Optional[PredictedTrajectory] = None,
    ):
        """
        场景中 Agent 的表示（车辆、行人、自行车和通用对象）。
        :param tracked_object_type: 当前 agent 的类型。
        :param oriented_box: Agent 的几何表示。
        :param velocity: Agent 的速度（矢量）。
        :param metadata: Agent 的元数据。
        :param angular_velocity: agent 的标量角速度（如果有）。
        :param predictions: 可选的（可能有多个）预测轨迹列表。
        :param past_trajectory: 此 agent 的可选过去轨迹。
        """
        AgentTemporalState.__init__(
            self,
            initial_time_stamp=TimePoint(metadata.timestamp_us),
            predictions=predictions,
            past_trajectory=past_trajectory,
        )
        AgentState.__init__(
            self,
            tracked_object_type=tracked_object_type,
            oriented_box=oriented_box,
            metadata=metadata,
            velocity=velocity,
            angular_velocity=angular_velocity,
        )

    @classmethod
    def from_agent_state(cls, agent: AgentState) -> Agent:
        """
        从 AgentState 创建 Agent。
        :param agent: 输入的单个 agent 状态。
        :return: 未来和过去轨迹均为 None 的 Agent。
        """
        return cls(
            tracked_object_type=agent.tracked_object_type,
            oriented_box=agent.box,
            velocity=agent.velocity,
            metadata=agent.metadata,
            angular_velocity=agent.angular_velocity,
            predictions=None,
            past_trajectory=None,
        )