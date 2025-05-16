from __future__ import annotations

import copy
from typing import Optional

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObject, SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType


class AgentState(SceneObject):
    """
    描述场景中 Agent 状态（包括动态）的类，表示车辆、自行车和行人。
    """

    def __init__(
        self,
        tracked_object_type: TrackedObjectType,
        oriented_box: OrientedBox,
        velocity: StateVector2D,
        metadata: SceneObjectMetadata,
        angular_velocity: Optional[float] = None,
    ):
        """
        场景中 Agent 的表示（车辆、行人、自行车和通用对象）。
        :param tracked_object_type: 当前 agent 的类型。
        :param oriented_box: Agent 的几何表示。
        :param velocity: Agent 的速度（矢量）。
        :param metadata: Agent 的元数据。
        :param angular_velocity: agent 的标量角速度（如果有）。
        """
        super().__init__(tracked_object_type=tracked_object_type, oriented_box=oriented_box, metadata=metadata)
        self._velocity = velocity
        self._angular_velocity = angular_velocity

    @property
    def velocity(self) -> StateVector2D:
        """
        获取速度。
        :return: Agent 的矢量速度。
        """
        return self._velocity

    @property
    def angular_velocity(self) -> Optional[float]:
        """
        获取角速度。
        :return: Agent 的角速度。
        """
        return self._angular_velocity

    @classmethod
    def from_new_pose(cls, agent: AgentState, pose: StateSE2) -> AgentState:
        """
        初始化一个具有新位姿的相同 agent。
        :param agent: 一个示例 agent。
        :param pose: 新的位姿。
        :return: 一个新的 agent。
        """
        return AgentState(
            tracked_object_type=agent.tracked_object_type,
            oriented_box=OrientedBox.from_new_pose(agent.box, pose),
            velocity=agent.velocity,
            angular_velocity=agent.angular_velocity,
            metadata=copy.deepcopy(agent.metadata),
        )