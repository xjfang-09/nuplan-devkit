from __future__ import annotations

from typing import Any, Iterable, List, Optional, Union

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.common.utils.split_state import SplitState


class Waypoint(InterpolatableState):
    """表示轨迹中的一个航点。选项允许表示几何轨迹"""

    def __init__(self, time_point: TimePoint, oriented_box: OrientedBox, velocity: Optional[StateVector2D] = None):
        """
        :param time_point: 航点对应的时间点
        :param oriented_box: 航点处的有向盒位置
        :param velocity: 可选的速度信息
        """
        self._time_point = time_point
        self._oriented_box = oriented_box
        self._velocity = velocity

    def __iter__(self) -> Iterable[Union[int, float]]:
        """
        航点变量的迭代器。
        :return: 航点变量的迭代器。
        """
        return iter(
            (
                self.time_us,
                self._oriented_box.center.x,
                self._oriented_box.center.y,
                self._oriented_box.center.heading,
                self._velocity.x if self._velocity is not None else None,
                self._velocity.y if self._velocity is not None else None,
            )
        )

    def __eq__(self, other: Any) -> bool:
        """
        比较两个航点是否相等。
        :param other: 另一个对象。
        :return: 如果两个对象相同则返回 True。
        """
        if not isinstance(other, Waypoint):
            return NotImplemented

        return (
            other.oriented_box == self._oriented_box
            and other.time_point == self.time_point
            and other.velocity == self._velocity
        )

    def __repr__(self) -> str:
        """
        :return: 描述对象的字符串。
        """
        return self.__class__.__qualname__ + "(" + ', '.join([f"{f}={v}" for f, v in self.__dict__.items()]) + ")"

    @property
    def center(self) -> StateSE2:
        """
        获取航点的中心位置
        :return: 表示航点位置的 StateSE2
        """
        return self._oriented_box.center

    @property
    def time_point(self) -> TimePoint:
        """
        获取航点对应的时间点
        :return: 时间点
        """
        return self._time_point

    @property
    def oriented_box(self) -> OrientedBox:
        """
        获取航点对应的有向盒
        :return: 有向盒
        """
        return self._oriented_box

    @property
    def x(self) -> float:
        """
        获取航点的 x 坐标
        :return: x 坐标
        """
        return self._oriented_box.center.x  # type:ignore

    @property
    def y(self) -> float:
        """
        获取航点的 y 坐标
        :return: y 坐标
        """
        return self._oriented_box.center.y  # type:ignore

    @property
    def heading(self) -> float:
        """
        获取航点的航向角
        :return: 航向角
        """
        return self._oriented_box.center.heading  # type:ignore

    @property
    def velocity(self) -> Optional[StateVector2D]:
        """
        获取航点对应的速度
        :return: 速度，如果不可用则为 None
        """
        return self._velocity

    def serialize(self) -> List[Union[int, float]]:
        """
        将对象序列化为列表
        :return: 序列化后的对象列表
        """
        return [
            self.time_point.time_us,
            self._oriented_box.center.x,
            self._oriented_box.center.y,
            self._oriented_box.center.heading,
            self._oriented_box.length,
            self._oriented_box.width,
            self._oriented_box.height,
            self._velocity.x if self._velocity is not None else None,
            self._velocity.y if self._velocity is not None else None,
        ]

    @staticmethod
    def deserialize(vector: List[Union[int, float]]) -> Waypoint:
        """
        反序列化对象。
        :param vector: 用于初始化航点的数据列表
        :return: 航点
        """
        assert len(vector) == 9, f'期望向量大小为 9，实际为 {len(vector)}'

        return Waypoint(
            time_point=TimePoint(int(vector[0])),
            oriented_box=OrientedBox(StateSE2(vector[1], vector[2], vector[3]), vector[4], vector[5], vector[6]),
            velocity=StateVector2D(vector[7], vector[8]) if vector[7] is not None and vector[8] is not None else None,
        )

    def to_split_state(self) -> SplitState:
        """继承自父类，参见父类文档。"""
        linear_states = [
            self.time_point.time_us,
            self._oriented_box.center.x,
            self._oriented_box.center.y,
            self._velocity.x if self._velocity is not None else None,
            self._velocity.y if self._velocity is not None else None,
        ]
        angular_states = [self._oriented_box.center.heading]
        fixed_state = [self._oriented_box.width, self._oriented_box.length, self._oriented_box.height]

        return SplitState(linear_states, angular_states, fixed_state)

    @staticmethod
    def from_split_state(split_state: SplitState) -> Waypoint:
        """继承自父类，参见父类文档。"""
        total_state_length = len(split_state)

        assert total_state_length == 9, f'期望向量大小为 9，实际为 {total_state_length}'

        return Waypoint(
            time_point=TimePoint(int(split_state.linear_states[0])),
            oriented_box=OrientedBox(
                StateSE2(split_state.linear_states[1], split_state.linear_states[2], split_state.angular_states[0]),
                length=split_state.fixed_states[1],
                width=split_state.fixed_states[0],
                height=split_state.fixed_states[2],
            ),
            velocity=StateVector2D(split_state.linear_states[3], split_state.linear_states[4])
            if split_state.linear_states[3] is not None and split_state.linear_states[4] is not None
            else None,
        )