import math
from typing import Optional

import torch


def _validate_state_se2_tensor_shape(tensor: torch.Tensor, expected_first_dim: Optional[int] = None) -> None:
    """
    验证一个张量是否符合 StateSE2 的形状要求。
    :param tensor: 要验证的张量。
    :param expected_first_dim: 第一维的期望值：
        * 1: 张量形状应为 (3,)
        * 2: 张量形状应为 (N, 3)
        * None: 两种形状都可以接受
    """
    expected_feature_dim = 3
    if len(tensor.shape) == 2 and tensor.shape[1] == expected_feature_dim:
        if expected_first_dim is None or expected_first_dim == 2:
            return
    if len(tensor.shape) == 1 and tensor.shape[0] == expected_feature_dim:
        if expected_first_dim is None or expected_first_dim == 1:
            return

    raise ValueError(f"不合法的 SE2 张量形状：{tensor.shape}")


def _validate_state_se2_tensor_batch_shape(tensor: torch.Tensor) -> None:
    """
    验证一个张量是否符合一批次 StateSE2 的形状要求。
    :param tensor: 要验证的张量。
    """
    expected_feature_dim = 3
    if len(tensor.shape) == 2 and tensor.shape[1] == expected_feature_dim:
        return

    raise ValueError(f"不合法的 SE2 批次张量形状：{tensor.shape}")


def _validate_transform_matrix_shape(tensor: torch.Tensor) -> None:
    """
    验证一个张量是否是 3x3 的变换矩阵。
    :param tensor: 要验证的张量。
    """
    if len(tensor.shape) == 2 and tensor.shape[0] == 3 and tensor.shape[1] == 3:
        return

    raise ValueError(f"不合法的变换矩阵形状：{tensor.shape}")


def _validate_transform_matrix_batch_shape(tensor: torch.Tensor) -> None:
    """
    验证一个张量是否是 Nx3x3 的变换矩阵批次。
    :param tensor: 要验证的张量。
    """
    if len(tensor.shape) == 3 and tensor.shape[1] == 3 and tensor.shape[2] == 3:
        return

    raise ValueError(f"不合法的变换矩阵批次形状：{tensor.shape}")


def state_se2_tensor_to_transform_matrix(
    input_data: torch.Tensor, precision: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    将形如 [x, y, heading] 的状态转换为 3x3 的变换矩阵。
    :param input_data: 输入数据（3维张量）。
    :return: 输出的 3x3 变换矩阵。
    """
    _validate_state_se2_tensor_shape(input_data, expected_first_dim=1)

    if precision is None:
        precision = input_data.dtype

    x: float = float(input_data[0].item())
    y: float = float(input_data[1].item())
    h: float = float(input_data[2].item())

    cosine: float = math.cos(h)
    sine: float = math.sin(h)

    return torch.tensor(
        [[cosine, -sine, x], [sine, cosine, y], [0.0, 0.0, 1.0]], dtype=precision, device=input_data.device
    )


def state_se2_tensor_to_transform_matrix_batch(
    input_data: torch.Tensor, precision: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    将 Nx3 形状的状态张量（x, y, heading）转换为 Nx3x3 的变换矩阵批次。
    :param input_data: 输入 Nx3 张量。
    :param precision: 输出张量的精度。如果为 None，则从输入中推断。
    :return: 输出的 Nx3x3 变换矩阵批次。
    """
    _validate_state_se2_tensor_batch_shape(input_data)

    if precision is None:
        precision = input_data.dtype

    processed_input = torch.column_stack(
        (
            input_data[:, 0],
            input_data[:, 1],
            torch.cos(input_data[:, 2]),
            torch.sin(input_data[:, 2]),
            torch.ones_like(input_data[:, 0], dtype=precision),
        )
    )

    reshaping_tensor = torch.tensor(
        [
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, -1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=precision,
        device=input_data.device,
    )

    return (processed_input @ reshaping_tensor).reshape(-1, 3, 3)


def transform_matrix_to_state_se2_tensor(
    input_data: torch.Tensor, precision: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    将 Nx3x3 的变换矩阵转换为 Nx3 的 [x, y, heading] 状态张量。
    :param input_data: Nx3x3 的变换矩阵。
    :param precision: 输出张量的精度。如果为 None，则从输入中推断。
    :return: 转换后的张量。
    """
    _validate_transform_matrix_shape(input_data)

    if precision is None:
        precision = input_data.dtype

    return torch.tensor(
        [
            float(input_data[0, 2].item()),
            float(input_data[1, 2].item()),
            float(math.atan2(float(input_data[1, 0].item()), float(input_data[0, 0].item())),
        ],
        dtype=precision,
    )


def transform_matrix_to_state_se2_tensor_batch(input_data: torch.Tensor) -> torch.Tensor:
    """
    将 Nx3x3 的变换矩阵批次转换为 Nx3 的 [x, y, heading] 状态张量。
    :param input_data: 3x3 的变换矩阵。
    :return: 转换后的张量。
    """
    _validate_transform_matrix_batch_shape(input_data)

    first_columns = input_data[:, :, 0].reshape(-1, 3)
    angles = torch.atan2(first_columns[:, 1], first_columns[:, 0])

    result = input_data[:, :, 2]
    result[:, 2] = angles

    return result


def global_state_se2_tensor_to_local(
    global_states: torch.Tensor, local_state: torch.Tensor, precision: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    将全局坐标系下的 StateSE2 张量转换到局部参考系。
    :param global_states: Nx3 的张量，列是 [x, y, heading]。
    :param local_state: [x, y, h] 格式的参考系。
    :param precision: 中间张量的精度。如果为 None，则从输入中推断。
    :return: 转换后的坐标。
    """
    _validate_state_se2_tensor_shape(global_states, expected_first_dim=2)
    _validate_state_se2_tensor_shape(local_state, expected_first_dim=1)

    if precision is None:
        if global_states.dtype != local_state.dtype:
            raise ValueError("传入 coordinates_to_local_frame 的数据类型混合，未指定精度。")
        precision = global_states.dtype

    local_xform = state_se2_tensor_to_transform_matrix(local_state, precision=precision)
    local_xform_inv = torch.linalg.inv(local_xform)

    transforms = state_se2_tensor_to_transform_matrix_batch(global_states, precision=precision)

    transforms = torch.matmul(local_xform_inv, transforms)

    output = transform_matrix_to_state_se2_tensor_batch(transforms)

    return output


def coordinates_to_local_frame(
    coords: torch.Tensor, anchor_state: torch.Tensor, precision: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    将一组 [x, y] 坐标（无航向角）转换到给定参考系下。
    :param coords: <torch.Tensor: num_coords, 2> 待转换的坐标，格式为 [x, y]。
    :param anchor_state: 目标参考系，格式为 [x, y, heading]。
    :param precision: 中间张量的精度。如果为 None，则从输入中推断。
    :return: <torch.Tensor: num_coords, 2> 转换后的坐标。
    """
    if len(coords.shape) != 2 or coords.shape[1] != 2:
        raise ValueError(f"意外的坐标形状：{coords.shape}")

    if precision is None:
        if coords.dtype != anchor_state.dtype:
            raise ValueError("传入 coordinates_to_local_frame 的数据类型混合，未指定精度。")
        precision = coords.dtype

    # 处理空输入
    if coords.shape[0] == 0:
        return coords

    transform = state_se2_tensor_to_transform_matrix(anchor_state, precision=precision)
    transform = torch.linalg.inv(transform)

    coords = torch.nn.functional.pad(coords, (0, 1, 0, 0), "constant", value=1.0)

    coords = torch.matmul(transform, coords.transpose(0, 1))

    result = coords.transpose(0, 1)
    result = result[:, :2]

    return result


def vector_set_coordinates_to_local_frame(
    coords: torch.Tensor,
    avails: torch.Tensor,
    anchor_state: torch.Tensor,
    output_precision: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    """
    将向量地图元素坐标从全局坐标系转换为自车坐标系（由 anchor_state 指定）。
    :param coords: 要转换的坐标。形状为 <torch.Tensor: num_elements, num_points, 2>。
    :param avails: 掩码标识真实数据与填充数据。形状为 <torch.Tensor: num_elements, num_points>。
    :param anchor_state: 目标坐标系，格式为 [x, y, heading]。
    :param output_precision: 输出张量的精度。
    :return: 转换后的坐标。
    :raise ValueError: 如果坐标维度无效或与 avail 不匹配。
    """
    if len(coords.shape) != 3 or coords.shape[2] != 2:
        raise ValueError(f"意外的坐标形状：{coords.shape}。期望形状：(*, *, 2)")

    if coords.shape[:2] != avails.shape:
        raise ValueError(f"坐标与可用性掩码形状不匹配：{coords.shape[:2]}, {avails.shape}")

    num_map_elements, num_points_per_element, _ = coords.size()
    coords = coords.reshape(num_map_elements * num_points_per_element, 2)

    coords = coordinates_to_local_frame(coords.double(), anchor_state.double(), precision=torch.float64)

    coords = coords.reshape(num_map_elements, num_points_per_element, 2)

    coords = coords.to(output_precision)

    coords[~avails] = 0.0

    return coords