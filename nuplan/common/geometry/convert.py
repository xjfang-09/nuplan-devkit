from typing import List

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D


def pose_from_matrix(transform_matrix: npt.NDArray[np.float32]) -> StateSE2:
    """
    将 3x3 的变换矩阵转换为一个 2D 姿态（StateSE2）。
    :param transform_matrix: 3x3 变换矩阵。
    :return: 2D 姿态 (x, y, yaw)
    """
    if transform_matrix.shape != (3, 3):
        raise RuntimeError(f"期望一个 3x3 的变换矩阵，实际形状为 {transform_matrix.shape}")

    heading = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])

    return StateSE2(transform_matrix[0, 2], transform_matrix[1, 2], heading)


def matrix_from_pose(pose: StateSE2) -> npt.NDArray[np.float64]:
    """
    将 2D 姿态（x, y, yaw）转换为一个 3x3 的变换矩阵。
    :param pose: 2D 姿态对象。
    :return: 3x3 变换矩阵。
    """
    return np.array(
        [
            [np.cos(pose.heading), -np.sin(pose.heading), pose.x],
            [np.sin(pose.heading), np.cos(pose.heading), pose.y],
            [0, 0, 1],
        ]
    )


def absolute_to_relative_poses(absolute_poses: List[StateSE2]) -> List[StateSE2]:
    """
    将一组 SE2 姿态从绝对坐标系转换为相对坐标系。
    第一个姿态作为参考原点（origin）。
    
    :param absolute_poses: 要转换的一组绝对姿态。
    :return: 转换后的相对姿态列表。
    """
    absolute_transforms: npt.NDArray[np.float64] = np.array([matrix_from_pose(pose) for pose in absolute_poses])
    origin_transform = np.linalg.inv(absolute_transforms[0])
    relative_transforms = origin_transform @ absolute_transforms
    relative_poses = [pose_from_matrix(transform_matrix) for transform_matrix in relative_transforms]

    return relative_poses


def relative_to_absolute_poses(origin_pose: StateSE2, relative_poses: List[StateSE2]) -> List[StateSE2]:
    """
    使用参考原点姿态将一组 SE2 相对姿态转换为绝对姿态。

    :param origin_pose: 参考原点姿态。
    :param relative_poses: 要转换的相对姿态列表。
    :return: 转换完成的绝对姿态列表。
    """
    relative_transforms: npt.NDArray[np.float64] = np.array([matrix_from_pose(pose) for pose in relative_poses])
    origin_transform = matrix_from_pose(origin_pose)
    absolute_transforms: npt.NDArray[np.float32] = origin_transform @ relative_transforms
    absolute_poses = [pose_from_matrix(transform_matrix) for transform_matrix in absolute_transforms]

    return absolute_poses


def numpy_array_to_absolute_velocity(
    origin_absolute_state: StateSE2, velocities: npt.NDArray[np.float32]
) -> List[StateVector2D]:
    """
    将一个包含相对速度的 NumPy 数组转换为绝对速度的 StateVector2D 列表。
    :param velocities: 要转换的速度数组。
    :param origin_absolute_state: 坐标转换的参考原点姿态。
    :return: 绝对速度矢量列表。
    """
    assert velocities.shape[1] == 2, f"期望输入形状为 (*, 2)，实际为 {velocities.shape}"
    velocities = np.pad(velocities.astype(np.float64), ((0, 0), (0, 1)), "constant", constant_values=0.0)
    relative_states = [StateSE2.deserialize(pose) for pose in velocities]
    return [
        StateVector2D(state.x, state.y) for state in relative_to_absolute_poses(origin_absolute_state, relative_states)
    ]


def numpy_array_to_absolute_pose(origin_absolute_state: StateSE2, poses: npt.NDArray[np.float32]) -> List[StateSE2]:
    """
    将一个包含相对姿态的 NumPy 数组转换为绝对姿态的 StateSE2 列表。
    :param poses: 要转换的姿态数组。
    :param origin_absolute_state: 姿态转换的参考原点。
    :return: 绝对姿态列表。
    """
    assert poses.shape[1] == 3, f"期望输入形状为 (*, 3)，实际为 {poses.shape}"
    relative_states = [StateSE2.deserialize(pose) for pose in poses]
    return relative_to_absolute_poses(origin_absolute_state, relative_states)


def vector_2d_from_magnitude_angle(magnitude: float, angle: float) -> StateVector2D:
    """
    将向量大小和角度投影为 x-y 分量组成的 2D 向量。
    :param magnitude: 向量大小。
    :param angle: 向量方向（弧度）。
    :return: 构建完成的 2D 向量（StateVector2D）。
    """
    return StateVector2D(np.cos(angle) * magnitude, np.sin(angle) * magnitude)