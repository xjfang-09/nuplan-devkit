from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Union

import numpy as np
import numpy.typing as npt


class TimeDuration:
    """表示时间间隔的类，具有微秒分辨率。"""

    __slots__ = "_time_us"

    def __init__(self, *, time_us: int, _direct: bool = True) -> None:
        """构造函数，不应直接调用。如果未设置关键字参数 _direct 为 False，则会引发异常。"""
        if _direct:
            raise RuntimeError("不要直接初始化此类，请使用构造函数！")

        self._time_us = time_us

    @classmethod
    def from_us(cls, t_us: int) -> TimeDuration:
        """
        从微秒值构造 TimeDuration。
        :param t_us: 时间（微秒）。
        :return: TimeDuration。
        """
        assert isinstance(t_us, int), "微秒值必须是整数！"
        return cls(time_us=t_us, _direct=False)

    @classmethod
    def from_ms(cls, t_ms: float) -> TimeDuration:
        """
        从毫秒值构造 TimeDuration。
        :param t_ms: 时间（毫秒）。
        :return: TimeDuration。
        """
        return cls(time_us=int(t_ms * int(1e3)), _direct=False)

    @classmethod
    def from_s(cls, t_s: float) -> TimeDuration:
        """
        从秒值构造 TimeDuration。
        :param t_s: 时间（秒）。
        :return: TimeDuration。
        """
        return cls(time_us=int(t_s * int(1e6)), _direct=False)

    @property
    def time_us(self) -> int:
        """
        :return: 时间间隔（微秒）。
        """
        return self._time_us

    @property
    def time_ms(self) -> float:
        """
        :return: 时间间隔（毫秒）。
        """
        return self._time_us / 1e3

    @property
    def time_s(self) -> float:
        """
        :return: 时间间隔（秒）。
        """
        return self._time_us / 1e6

    def __add__(self, other: object) -> TimeDuration:
        """
        将时间间隔相加。
        :param other: 时间间隔。
        :return: self + other（如果 other 是 TimeDuration）。
        """
        if isinstance(other, TimeDuration):
            return TimeDuration.from_us(self.time_us + other.time_us)
        return NotImplemented

    def __sub__(self, other: object) -> TimeDuration:
        """
        从时间间隔中减去另一个时间间隔。
        :param other: 时间间隔。
        :return: self - other（如果 other 是 TimeDuration）。
        """
        if isinstance(other, TimeDuration):
            return TimeDuration.from_us(self.time_us - other.time_us)
        return NotImplemented

    def __mul__(self, other: object) -> TimeDuration:
        """
        将时间间隔乘以一个标量值。
        :param other: 要乘以的值。
        :return: self * other（如果 other 是标量）。
        """
        if isinstance(other, (int, float)):
            return TimeDuration.from_s(self.time_s * other)
        return NotImplemented

    def __rmul__(self, other: object) -> TimeDuration:
        """
        将时间间隔乘以一个标量值。
        :param other: 要乘以的值。
        :return: self * other（如果 other 是标量）。
        """
        if isinstance(other, (int, float)):
            return self * other
        return NotImplemented

    def __truediv__(self, other: object) -> TimeDuration:
        """
        将时间间隔除以一个标量值。
        :param other: 要除以的值。
        :return: self / other（如果 other 是标量）。
        """
        if isinstance(other, (int, float)):
            return TimeDuration.from_s(self.time_s / other)
        return NotImplemented

    def __floordiv__(self, other: object) -> TimeDuration:
        """
        将时间间隔整除一个标量值。
        :param other: 要整除的值。
        :return: self // other（如果 other 是标量）。
        """
        if isinstance(other, (int, float)):
            return TimeDuration.from_s(self.time_s // other)
        return NotImplemented

    def __gt__(self, other: TimeDuration) -> bool:
        """
        判断 self 是否大于 other。
        :param other: 时间间隔。
        :return: 如果 self > other，则返回 True，否则返回 False。
        """
        if isinstance(other, TimeDuration):
            return self.time_us > other.time_us
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        """
        判断 self 是否大于或等于 other。
        :param other: 时间间隔。
        :return: 如果 self >= other，则返回 True，否则返回 False。
        """
        if isinstance(other, TimeDuration):
            return self.time_us >= other.time_us
        return NotImplemented

    def __lt__(self, other: TimeDuration) -> bool:
        """
        判断 self 是否小于 other。
        :param other: 时间间隔。
        :return: 如果 self < other，则返回 True，否则返回 False。
        """
        if isinstance(other, TimeDuration):
            return self.time_us < other.time_us
        return NotImplemented

    def __le__(self, other: TimeDuration) -> bool:
        """
        判断 self 是否小于或等于 other。
        :param other: 时间间隔。
        :return: 如果 self <= other，则返回 True，否则返回 False。
        """
        if isinstance(other, TimeDuration):
            return self.time_us <= other.time_us
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        """
        判断 self 是否等于 other。
        :param other: 时间间隔。
        :return: 如果 self == other，则返回 True，否则返回 False。
        """
        if not isinstance(other, TimeDuration):
            return NotImplemented

        return self.time_us == other.time_us

    def __hash__(self) -> int:
        """
        :return: 此对象的哈希值。
        """
        return hash(self.time_us)

    def __repr__(self) -> str:
        """
        :return: 字符串表示。
        """
        return "TimeDuration({}s)".format(self.time_s)


@dataclass
class TimePoint:
    """
    时间序列中的时间点实例。
    """

    time_us: int  # [微秒] 自纪元以来的微秒数
    __slots__ = "time_us"

    def __post_init__(self) -> None:
        """
        创建后验证类。
        """
        assert self.time_us >= 0, "时间点必须为正数！"

    @property
    def time_s(self) -> float:
        """
        :return [秒] 时间（秒）。
        """
        return self.time_us * 1e-6

    def __add__(self, other: object) -> TimePoint:
        """
        将一个 TimeDuration 添加到生成一个新的 TimePoint。
        :param other: 时间点。
        :return: self + other。
        """
        if isinstance(other, (TimeDuration, TimePoint)):
            return TimePoint(self.time_us + other.time_us)
        return NotImplemented

    def __radd__(self, other: object) -> TimePoint:
        """
        :param other: 右加操作的目标。
        :return: 如果 other 是 TimeDuration，则返回加法结果。
        """
        if isinstance(other, TimeDuration):
            return self.__add__(other)
        return NotImplemented

    def __sub__(self, other: object) -> TimePoint:
        """
        从一个时间点减去一个时间间隔。
        :param other: 时间间隔。
        :return: 如果 other 是 TimeDuration，则返回 self - other。
        """
        if isinstance(other, (TimeDuration, TimePoint)):
            return TimePoint(self.time_us - other.time_us)
        return NotImplemented

    def __gt__(self, other: TimePoint) -> bool:
        """
        判断 self 是否大于 other。
        :param other: 时间点。
        :return: 如果 self > other，则返回 True，否则返回 False。
        """
        if isinstance(other, TimePoint):
            return self.time_us > other.time_us
        return NotImplemented

    def __ge__(self, other: TimePoint) -> bool:
        """
        判断 self 是否大于或等于 other。
        :param other: 时间点。
        :return: 如果 self >= other，则返回 True，否则返回 False。
        """
        if isinstance(other, TimePoint):
            return self.time_us >= other.time_us
        return NotImplemented

    def __lt__(self, other: TimePoint) -> bool:
        """
        判断 self 是否小于 other。
        :param other: 时间点。
        :return: 如果 self < other，则返回 True，否则返回 False。
        """
        if isinstance(other, TimePoint):
            return self.time_us < other.time_us
        return NotImplemented

    def __le__(self, other: TimePoint) -> bool:
        """
        判断 self 是否小于或等于 other。
        :param other: 时间点。
        :return: 如果 self <= other，则返回 True，否则返回 False。
        """
        if isinstance(other, TimePoint):
            return self.time_us <= other.time_us
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        """
        判断 self 是否等于 other。
        :param other: 时间点。
        :return: 如果 self == other，则返回 True，否则返回 False。
        """
        if not isinstance(other, TimePoint):
            return NotImplemented

        return self.time_us == other.time_us

    def __hash__(self) -> int:
        """
        :return: 此对象的哈希值。
        """
        return hash(self.time_us)

    def diff(self, time_point: TimePoint) -> TimeDuration:
        """
        计算 self 和另一个 TimePoint 之间的 TimeDuration。
        :param time_point: 另一个时间点。
        :return: 两个时间点之间的 TimeDuration。
        """
        return TimeDuration.from_us(int(self.time_us - time_point.time_us))


@dataclass
class Point2D:
    """表示二维点的类。"""

    x: float  # [米] 位置
    y: float  # [米] 位置
    __slots__ = "x", "y"

    def __iter__(self) -> Iterable[float]:
        """
        :return: (x, y) 的迭代器。
        """
        return iter((self.x, self.y))

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """
        将向量转换为数组。
        :return: 包含 [x, y] 的数组。
        """
        return np.array([self.x, self.y], dtype=np.float64)

    def __hash__(self) -> int:
        """哈希方法。"""
        return hash((self.x, self.y))


@dataclass
class StateSE2(Point2D):
    """
    SE2 状态 - 表示 [x, y, heading]
    """

    heading: float  # [弧度] 状态的航向角
    __slots__ = "heading"

    @property
    def point(self) -> Point2D:
        """
        从 StateSE2 获取一个点
        :return: 包含 x 和 y 的 Point2D
        """
        return Point2D(self.x, self.y)

    def as_matrix(self) -> npt.NDArray[np.float32]:
        """
        :return: 表示 SE2 状态的 3x3 二维变换矩阵。
        """
        return np.array(
            [
                [np.cos(self.heading), -np.sin(self.heading), self.x],
                [np.sin(self.heading), np.cos(self.heading), self.y],
                [0.0, 0.0, 1.0],
            ]
        )

    def as_matrix_3d(self) -> npt.NDArray[np.float32]:
        """
        :return: 表示投影到 SE3 的 SE2 状态的 4x4 三维变换矩阵。
        """
        return np.array(
            [
                [np.cos(self.heading), -np.sin(self.heading), 0.0, self.x],
                [np.sin(self.heading), np.cos(self.heading), 0.0, self.y],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def distance_to(self, state: StateSE2) -> float:
        """
        计算两个点之间的欧几里得距离
        :param state: 要计算距离的状态
        :return: 两个点之间的距离
        """
        return float(np.hypot(self.x - state.x, self.y - state.y))

    @staticmethod
    def from_matrix(matrix: npt.NDArray[np.float32]) -> StateSE2:
        """
        :param matrix: 3x3 二维变换矩阵
        :return: StateSE2 对象
        """
        assert matrix.shape == (3, 3), f"期望 3x3 变换矩阵，但输入矩阵的形状为 {matrix.shape}"

        vector = [matrix[0, 2], matrix[1, 2], np.arctan2(matrix[1, 0], matrix[0, 0])]
        return StateSE2.deserialize(vector)

    @staticmethod
    def deserialize(vector: List[float]) -> StateSE2:
        """
        将向量反序列化为 SE2 状态
        :param vector: 序列化的浮点数列表
        :return: StateSE2
        """
        if len(vector) != 3:
            raise RuntimeError(f'期望向量大小为 3，实际为 {len(vector)}')

        return StateSE2(x=vector[0], y=vector[1], heading=vector[2])

    def serialize(self) -> List[float]:
        """
        :return: 序列化变量的列表 [X, Y, Heading]
        """
        return [self.x, self.y, self.heading]

    def __eq__(self, other: object) -> bool:
        """
        比较两个 SE2 状态
        :param other: 对象
        :return: 如果对象相等则返回 True，否则返回 False
        """
        if not isinstance(other, StateSE2):
            # 如果类类型不同，返回 NotImplemented
            return NotImplemented
        return (
            math.isclose(self.x, other.x, abs_tol=1e-3)
            and math.isclose(self.y, other.y, abs_tol=1e-3)
            and math.isclose(self.heading, other.heading, abs_tol=1e-4)
        )

    def __iter__(self) -> Iterable[float]:
        """
        :return: 包含 (x, y, heading) 的迭代器
        """
        return iter((self.x, self.y, self.heading))

    def __hash__(self) -> int:
        """
        :return: 此对象的哈希值
        """
        return hash((self.x, self.y, self.heading))


@dataclass
class ProgressStateSE2(StateSE2):
    """
    通过进度参数化的 SE2 状态
    """

    progress: float  # [米] 沿路径的距离
    __slots__ = "progress"

    @staticmethod
    def deserialize(vector: List[float]) -> ProgressStateSE2:
        """
        将向量反序列化为此类
        :param vector: 包含原始浮点数 [progress, x, y, heading]
        :return: ProgressStateSE2 类
        """
        if len(vector) != 4:
            raise RuntimeError(f'期望向量大小为 4，实际为 {len(vector)}')

        return ProgressStateSE2(progress=vector[0], x=vector[1], y=vector[2], heading=vector[3])

    def __iter__(self) -> Iterable[Union[float]]:
        """
        :return: 包含 (progress, x, y, heading) 状态的迭代器
        """
        return iter((self.progress, self.x, self.y, self.heading))


@dataclass
class TemporalStateSE2(StateSE2):
    """
    表示时间状态的类
    """

    time_point: TimePoint  # 状态对应的时间点

    @property
    def time_us(self) -> int:
        """
        :return: [微秒] 时间戳
        """
        return self.time_point.time_us

    @property
    def time_seconds(self) -> float:
        """
        :return: [秒] 时间戳
        """
        return self.time_us * 1e-6


class StateVector2D:
    """表示二维向量的类。"""

    __slots__ = "_x", "_y", "_array"

    def __init__(self, x: float, y: float):
        """
        创建 StateVector2D 对象
        :param x: 浮点数方向
        :param y: 浮点数方向
        """
        self._x = x  # 向量的 x 轴。
        self._y = y  # 向量的 y 轴。

        self._array: npt.NDArray[np.float64] = np.array([self.x, self.y], dtype=np.float64)

    def __repr__(self) -> str:
       # filepath: /home/mark/nuplan-devkit/nuplan/common/actor_state/state_representation.py
@dataclass
class StateSE2(Point2D):
    """
    SE2 状态 - 表示 [x, y, heading]
    """

    heading: float  # [弧度] 状态的航向角
    __slots__ = "heading"

    @property
    def point(self) -> Point2D:
        """
        从 StateSE2 获取一个点
        :return: 包含 x 和 y 的 Point2D
        """
        return Point2D(self.x, self.y)

    def as_matrix(self) -> npt.NDArray[np.float32]:
        """
        :return: 表示 SE2 状态的 3x3 二维变换矩阵。
        """
        return np.array(
            [
                [np.cos(self.heading), -np.sin(self.heading), self.x],
                [np.sin(self.heading), np.cos(self.heading), self.y],
                [0.0, 0.0, 1.0],
            ]
        )

    def as_matrix_3d(self) -> npt.NDArray[np.float32]:
        """
        :return: 表示投影到 SE3 的 SE2 状态的 4x4 三维变换矩阵。
        """
        return np.array(
            [
                [np.cos(self.heading), -np.sin(self.heading), 0.0, self.x],
                [np.sin(self.heading), np.cos(self.heading), 0.0, self.y],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def distance_to(self, state: StateSE2) -> float:
        """
        计算两个点之间的欧几里得距离
        :param state: 要计算距离的状态
        :return: 两个点之间的距离
        """
        return float(np.hypot(self.x - state.x, self.y - state.y))

    @staticmethod
    def from_matrix(matrix: npt.NDArray[np.float32]) -> StateSE2:
        """
        :param matrix: 3x3 二维变换矩阵
        :return: StateSE2 对象
        """
        assert matrix.shape == (3, 3), f"期望 3x3 变换矩阵，但输入矩阵的形状为 {matrix.shape}"

        vector = [matrix[0, 2], matrix[1, 2], np.arctan2(matrix[1, 0], matrix[0, 0])]
        return StateSE2.deserialize(vector)

    @staticmethod
    def deserialize(vector: List[float]) -> StateSE2:
        """
        将向量反序列化为 SE2 状态
        :param vector: 序列化的浮点数列表
        :return: StateSE2
        """
        if len(vector) != 3:
            raise RuntimeError(f'期望向量大小为 3，实际为 {len(vector)}')

        return StateSE2(x=vector[0], y=vector[1], heading=vector[2])

    def serialize(self) -> List[float]:
        """
        :return: 序列化变量的列表 [X, Y, Heading]
        """
        return [self.x, self.y, self.heading]

    def __eq__(self, other: object) -> bool:
        """
        比较两个 SE2 状态
        :param other: 对象
        :return: 如果对象相等则返回 True，否则返回 False
        """
        if not isinstance(other, StateSE2):
            # 如果类类型不同，返回 NotImplemented
            return NotImplemented
        return (
            math.isclose(self.x, other.x, abs_tol=1e-3)
            and math.isclose(self.y, other.y, abs_tol=1e-3)
            and math.isclose(self.heading, other.heading, abs_tol=1e-4)
        )

    def __iter__(self) -> Iterable[float]:
        """
        :return: 包含 (x, y, heading) 的迭代器
        """
        return iter((self.x, self.y, self.heading))

    def __hash__(self) -> int:
        """
        :return: 此对象的哈希值
        """
        return hash((self.x, self.y, self.heading))


@dataclass
class ProgressStateSE2(StateSE2):
    """
    通过进度参数化的 SE2 状态
    """

    progress: float  # [米] 沿路径的距离
    __slots__ = "progress"

    @staticmethod
    def deserialize(vector: List[float]) -> ProgressStateSE2:
        """
        将向量反序列化为此类
        :param vector: 包含原始浮点数 [progress, x, y, heading]
        :return: ProgressStateSE2 类
        """
        if len(vector) != 4:
            raise RuntimeError(f'期望向量大小为 4，实际为 {len(vector)}')

        return ProgressStateSE2(progress=vector[0], x=vector[1], y=vector[2], heading=vector[3])

    def __iter__(self) -> Iterable[Union[float]]:
        """
        :return: 包含 (progress, x, y, heading) 状态的迭代器
        """
        return iter((self.progress, self.x, self.y, self.heading))


@dataclass
class TemporalStateSE2(StateSE2):
    """
    表示时间状态的类
    """

    time_point: TimePoint  # 状态对应的时间点

    @property
    def time_us(self) -> int:
        """
        :return: [微秒] 时间戳
        """
        return self.time_point.time_us

    @property
    def time_seconds(self) -> float:
        """
        :return: [秒] 时间戳
        """
        return self.time_us * 1e-6
class StateVector2D:
    """表示二维向量的类。"""

    __slots__ = "_x", "_y", "_array"

    def __init__(self, x: float, y: float):
        """
        创建 StateVector2D 对象
        :param x: 浮点数方向
        :param y: 浮点数方向
        """
        self._x = x  # 向量的 x 轴。
        self._y = y  # 向量的 y 轴。

        self._array: npt.NDArray[np.float64] = np.array([self.x, self.y], dtype=np.float64)

    def __repr__(self) -> str:
        """
        :return: 包含此类表示的字符串
        """
        return f'x: {self.x}, y: {self.y}'

    def __eq__(self, other: object) -> bool:
        """
        比较其他对象与此类是否相等
        :param other: 对象
        :return: 如果其他状态向量与 self 相同，则返回 True
        """
        if not isinstance(other, StateVector2D):
            return NotImplemented
        return bool(np.array_equal(self.array, other.array))

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """
        将向量转换为数组
        :return: 包含 [x, y] 的数组
        """
        return self._array

    @array.setter
    def array(self, other: npt.NDArray[np.float64]) -> None:
        """自定义 setter，以防止对象被破坏。"""
        self._array = other
        self._x = other[0]
        self._y = other[1]

    @property
    def x(self) -> float:
        """
        :return: x 浮点状态
        """
        return self._x

    @x.setter
    def x(self, x: float) -> None:
        """自定义 setter，以防止对象被破坏。"""
        self._x = x
        self._array[0] = x

    @property
    def y(self) -> float:
        """
        :return: y 浮点状态
        """
        return self._y

    @y.setter
    def y(self, y: float) -> None:
        """自定义 setter，以防止对象被破坏。"""
        self._y = y
        self._array[1] = y

    def magnitude(self) -> float:
        """
        :return: 向量的大小
        """
        return float(np.hypot(self.x, self.y))