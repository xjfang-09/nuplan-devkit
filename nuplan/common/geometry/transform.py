import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import Point2D, StateSE2


def rotate_2d(point: Point2D, rotation_matrix: npt.NDArray[np.float64]) -> Point2D:
    """
    使用二维旋转矩阵对一个2D点进行旋转。
    :param point: 需要被旋转的点
    :param rotation_matrix: 旋转矩阵 [[R11, R12], [R21, R22]]
    :return: 旋转后的点
    """
    assert rotation_matrix.shape == (2, 2)
    rotated_point = np.array([point.x, point.y]) @ rotation_matrix
    return Point2D(rotated_point[0], rotated_point[1])


def translate(pose: StateSE2, translation: npt.NDArray[np.float64]) -> StateSE2:
    """
    应用二维平移
    :param pose: 被变换的姿态
    :param translation: 要应用的平移量
    :return: 平移后的新姿态
    """
    assert translation.shape == (2,) or translation.shape == (2, 1)
    return StateSE2(pose.x + translation[0], pose.y + translation[1], pose.heading)


def rotate(pose: StateSE2, rotation_matrix: npt.NDArray[np.float64]) -> StateSE2:
    """
    对 SE2 姿态应用二维旋转
    :param pose: 需要被变换的姿态
    :param rotation_matrix: 表示旋转的 2x2 旋转矩阵
    :return: 旋转后的新姿态
    """
    assert rotation_matrix.shape == (2, 2)
    rotated_point = np.array([pose.x, pose.y]) @ rotation_matrix
    rotation_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[1, 1])
    return StateSE2(rotated_point[0], rotated_point[1], pose.heading + rotation_angle)


def rotate_angle(pose: StateSE2, theta: float) -> StateSE2:
    """
    按给定角度旋转场景对象
    :param pose: 输入姿态
    :param theta: 旋转角度
    """
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix: npt.NDArray[np.float64] = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    return rotate(pose, rotation_matrix)


def transform(pose: StateSE2, transform_matrix: npt.NDArray[np.float64]) -> StateSE2:
    """
    应用 SE2 变换
    :param pose: 输入姿态
    :param transform_matrix: 变换矩阵，可以是二维（3x3）或三维（4x4）
    """
    rotated_pose = rotate(pose, transform_matrix[:2, :2])
    return translate(rotated_pose, transform_matrix[:2, 2])


def translate_longitudinally(pose: StateSE2, distance: float) -> StateSE2:
    """
    沿航向方向对 SE2 姿态进行纵向平移
    :param pose: 要被平移的 SE2 姿态
    :param distance: [米] 纵向平移距离
    :return: 平移后的 SE2 姿态
    """
    translation: npt.NDArray[np.float64] = np.array([distance * np.cos(pose.heading), distance * np.sin(pose.heading)])
    return translate(pose, translation)


def translate_laterally(pose: StateSE2, distance: float) -> StateSE2:
    """
    对 SE2 姿态进行横向平移
    :param pose: 要被平移的 SE2 姿态
    :param distance: [米] 横向平移距离
    :return: 平移后的 SE2 姿态
    """
    half_pi = np.pi / 2.0
    translation: npt.NDArray[np.float64] = np.array(
        [distance * np.cos(pose.heading + half_pi), distance * np.sin(pose.heading + half_pi)]
    )
    return translate(pose, translation)


def translate_longitudinally_and_laterally(pose: StateSE2, lon: float, lat: float) -> StateSE2:
    """
    对 SE2 姿态的位置分量进行纵向和横向平移
    :param pose: 要被平移的 SE2 姿态
    :param lon: [米] 纵向平移距离
    :param lat: [米] 横向平移距离
    :return: 平移后的二维位置
    """
    half_pi = np.pi / 2.0
    translation: npt.NDArray[np.float64] = np.array(
        [
            (lat * np.cos(pose.heading + half_pi)) + (lon * np.cos(pose.heading)),
            (lat * np.sin(pose.heading + half_pi)) + (lon * np.sin(pose.heading)),
        ]
    )
    return translate(pose, translation)