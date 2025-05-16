from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObject, SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateVector2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType


class StaticObject(SceneObject):
    """表示场景中的静态对象。"""

    def __init__(
        self, tracked_object_type: TrackedObjectType, oriented_box: OrientedBox, metadata: SceneObjectMetadata
    ):
        """
        :param tracked_object_type: 对象的分类类型。
        :param oriented_box: 几何上表示静态对象的 OrientedBox。
        :param metadata: 静态对象的元数据。
        """
        super().__init__(tracked_object_type, oriented_box, metadata)

        # TODO: 这些字段可以在检查它们的访问方式后移除
        self.predictions = None
        self.past_trajectory = None
        self.velocity = StateVector2D(0.0, 0.0)