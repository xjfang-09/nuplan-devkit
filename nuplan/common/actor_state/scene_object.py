from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType


@dataclass(frozen=True)
class SceneObjectMetadata:
    """
    场景中每个对象的元数据。
    """

    # 时间戳（微秒）
    timestamp_us: int
    # 全局唯一的 token
    token: str
    # 对象可读 ID
    track_id: Optional[int]
    # 时序一致的对象 token
    track_token: Optional[str]
    # 类别名称（可读字符串）
    category_name: Optional[str] = None

    @property
    def timestamp_s(self) -> float:
        """
        :return: 时间戳（以秒为单位）。
        """
        return self.timestamp_us * 1e-6


class SceneObject:
    """
    表示场景中的对象，如车辆、行人等。
    """

    def __init__(
        self,
        tracked_object_type: TrackedObjectType,
        oriented_box: OrientedBox,
        metadata: SceneObjectMetadata,
    ):
        """
        初始化一个场景对象。
        :param tracked_object_type: 当前静态对象的类型。
        :param oriented_box: 静态对象的几何表示。
        :param metadata: 对象的高层信息。
        """
        self._metadata = metadata
        self.instance_token = None
        self._tracked_object_type = tracked_object_type
        self._box: OrientedBox = oriented_box

    @property
    def metadata(self) -> SceneObjectMetadata:
        """
        获取对象的元数据。
        :return: SceneObjectMetadata 实例。
        """
        return self._metadata

    @property
    def token(self) -> str:
        """
        获取对象唯一 token。不同样本中同一对象的 token 不同。
        :return: 唯一 token。
        """
        return self._metadata.token

    @property
    def track_token(self) -> Optional[str]:
        """
        获取跨样本跟踪的唯一 token。同一对象在不同样本中 track_token 相同。
        :return: 唯一 track_token。
        """
        return self._metadata.track_token

    @property
    def tracked_object_type(self) -> TrackedObjectType:
        """
        获取对象分类类型。
        :return: TrackedObjectType 实例。
        """
        return self._tracked_object_type

    @property
    def box(self) -> OrientedBox:
        """
        获取对象的 OrientedBox 几何表示。
        :return: OrientedBox 实例。
        """
        return self._box

    @property
    def center(self) -> StateSE2:
        """
        获取对象的中心姿态。
        :return: StateSE2 实例。
        """
        return self.box.center

    @classmethod
    def make_random(cls, token: str, object_type: TrackedObjectType) -> SceneObject:
        """
        创建一个随机生成的 SceneObject 实例。
        :param token: 唯一标识 token。
        :param object_type: 分类类型。
        :return: SceneObject 实例。
        """
        center = random.sample(range(50), 2)
        heading = np.random.uniform(-np.pi, np.pi)
        size = random.sample(range(1, 50), 3)
        track_id = random.sample(range(1, 10), 1)[0]
        timestamp_us = random.sample(range(1, 10), 1)[0]

        return SceneObject(
            metadata=SceneObjectMetadata(token=token, track_id=track_id, track_token=token, timestamp_us=timestamp_us),
            tracked_object_type=object_type,
            oriented_box=OrientedBox(StateSE2(*center, heading), size[0], size[1], size[2]),
        )

    @classmethod
    def from_raw_params(
        cls,
        token: str,
        track_token: str,
        timestamp_us: int,
        track_id: int,
        center: StateSE2,
        size: Tuple[float, float, float],
    ) -> SceneObject:
        """
        根据原始参数创建通用场景对象。
        :param token: 对象的 token。
        :param track_token: 跨样本跟踪的 token。
        :param timestamp_us: [us] 对象的时间戳。
        :param track_id: 可读的对象 ID。
        :param center: 对象的中心姿态。
        :param size: 几何包围盒大小 (width, length, height)。
        :return: SceneObject 实例。
        """
        box = OrientedBox(center, width=size[0], length=size[1], height=size[2])
        return SceneObject(
            metadata=SceneObjectMetadata(
                token=token, track_token=track_token, timestamp_us=timestamp_us, track_id=track_id
            ),
            tracked_object_type=TrackedObjectType.GENERIC_OBJECT,
            oriented_box=box,
        )