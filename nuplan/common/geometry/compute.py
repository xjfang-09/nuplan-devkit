from typing import List, Union

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d
from shapely.geometry import Polygon

from nuplan.common.actor_state.oriented_box import Dimension, OrientedBox
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.geometry.transform import translate_laterally, translate_longitudinally


def lateral_distance(reference: StateSE2, other: Point2D) -> float:
    """
    计算从一个点到参考位姿的横向距离
    :param reference: 参考位姿
    :param other: 查询点
    :return: 横向距离
    """
    return float(
        -np.sin(reference.heading) * (other.x - reference.x) + np.cos(reference.heading) * (other.y - reference.y)
    )


def longitudinal_distance(reference: StateSE2, other: Point2D) -> float:
    """
    计算从一个点到参考位姿的纵向距离
    :param reference: 参考位姿
    :param other: 查询点
    :return: 纵向距离
    """
    return float(
        np.cos(reference.heading) * (other.x - reference.x) + np.sin(reference.heading) * (other.y - reference.y)
    )


def signed_lateral_distance(ego_state: StateSE2, other: Polygon) -> float:
    """
    计算自车与另一个多边形之间的最小横向距离
    :param ego_state: 自车的状态
    :param other: 查询多边形
    :return: 带符号的横向距离
    """
    ego_half_width = get_pacifica_parameters().half_width
    ego_left = translate_laterally(ego_state, ego_half_width)
    ego_right = translate_laterally(ego_state, -ego_half_width)

    vertices = list(zip(*other.exterior.coords.xy))
    distance_left = max(min(lateral_distance(ego_left, Point2D(*vertex)) for vertex in vertices), 0)
    distance_right = max(min(-lateral_distance(ego_right, Point2D(*vertex)) for vertex in vertices), 0)
    return distance_left if distance_left > distance_right else -distance_right


def signed_longitudinal_distance(ego_state: StateSE2, other: Polygon) -> float:
    """
    计算自车与另一个多边形之间的最小纵向距离
    :param ego_state: 自车的状态
    :param other: 查询多边形
    :return: 带符号的纵向距离
    """
    ego_half_length = get_pacifica_parameters().half_length
    ego_front = translate_longitudinally(ego_state, ego_half_length)
    ego_back = translate_longitudinally(ego_state, -ego_half_length)

    vertices = list(zip(*other.exterior.coords.xy))
    distance_front = max(min(longitudinal_distance(ego_front, Point2D(*vertex)) for vertex in vertices), 0)
    distance_back = max(min(-longitudinal_distance(ego_back, Point2D(*vertex)) for vertex in vertices), 0)
    return distance_front if distance_front > distance_back else -distance_back


def compute_distance(lhs: StateSE2, rhs: StateSE2) -> float:
    """
    计算两点之间的欧几里得距离
    :param lhs: 第一个点
    :param rhs: 第二个点
    :return: 两点之间的距离
    """
    return float(np.hypot(lhs.x - rhs.x, lhs.y - rhs.y))


def compute_lateral_displacements(poses: List[StateSE2]) -> List[float]:
    """
    计算一组位姿的横向位移 (y_t - y_t-1)

    :param poses: 用于计算位移的 N 个位姿列表
    :return: N-1 个横向位移的列表
    """
    return [poses[idx].y - poses[idx - 1].y for idx in range(1, len(poses))]


def principal_value(
    angle: Union[float, int, npt.NDArray[np.float64]], min_: float = -np.pi
) -> Union[float, npt.NDArray[np.float64]]:
    """
    将角度限制在指定的范围内 (2π 的倍数)，确保角度在 [min_, min_ + 2π) 范围内。
    如果角度为无穷大，则抛出错误。
    :param angle: 弧度
    :param min_: 角度的最小范围 (弧度)
    :return: 限制在 [min_, min_ + 2π) 范围内的角度
    """
    assert np.all(np.isfinite(angle)), "角度不是有限值"

    lhs = (angle - min_) % (2 * np.pi) + min_

    return lhs


def l2_euclidean_corners_distance(box1: OrientedBox, box2: OrientedBox) -> float:
    """
    计算两个 OrientedBox 的角点之间欧几里得距离的 L2 范数 [m]。
    :param box1: 第一个盒子的配置
    :param box2: 第二个盒子的配置
    :return: 欧几里得距离的范数 [m]
    """
    distances = [
        np.linalg.norm(box1_corner.array - box2_corner.array)
        for box1_corner, box2_corner in zip(box1.all_corners(), box2.all_corners())
    ]
    return float(np.linalg.norm(distances))


def se2_box_distances(
    query: StateSE2, targets: list[StateSE2], box_size: Dimension, consider_flipped: bool = True
) -> List[float]:
    """
    计算从查询点到目标点列表的最小距离 [m]。
    距离通过计算以位姿为中心并具有给定尺寸的盒子的角点之间的欧几里得距离的范数来计算。
    查询盒子还会旋转 180 度，并使用两者中的最小距离。
    :param query: 查询位姿
    :param targets: 用于计算距离的目标
    :param box_size: 要构造的盒子的尺寸
    :param consider_flipped: 是否检查查询位姿旋转 180 度的情况
    :return: 从查询点到目标点的距离列表 [m]
    """
    query_box = OrientedBox(query, box_size.length, box_size.width, box_size.height)
    backwards_query_box = OrientedBox.from_new_pose(query_box, StateSE2(query.x, query.y, query.heading + np.pi))
    target_boxes = [OrientedBox(target, box_size.length, box_size.width, box_size.height) for target in targets]
    if consider_flipped:
        return [
            min(
                l2_euclidean_corners_distance(query_box, target_box),
                l2_euclidean_corners_distance(backwards_query_box, target_box),
            )
            for target_box in target_boxes
        ]
    else:
        return [l2_euclidean_corners_distance(query_box, target_box) for target_box in target_boxes]


class AngularInterpolator:
    """创建一个角度线性插值器。"""

    def __init__(self, states: npt.NDArray[np.float64], angular_states: npt.NDArray[np.float64]):
        """
        :param states: 用于插值的 x 值
        :param angular_states: 用于插值的 y 值
        """
        _angular_states = np.unwrap(angular_states, axis=0)

        self.interpolator = interp1d(states, _angular_states, axis=0)

    def interpolate(self, sampled_state: Union[float, List[float]]) -> npt.NDArray[np.float64]:
        """
        对单个状态进行插值
        :param sampled_state: 要执行插值的状态
        :return: 在给定状态下线性插值的状态值
        """
        return principal_value(self.interpolator(sampled_state))  # type: ignore