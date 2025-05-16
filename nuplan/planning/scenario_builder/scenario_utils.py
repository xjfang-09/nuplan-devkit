from typing import List


def sample_indices_with_time_horizon(num_samples: int, time_horizon: float, time_interval: float) -> List[int]:
    """
    从一个时间间隔为 DT 的时序序列中，采样能在 T 时间范围内获取 N 个样本的索引。
    :param num_samples: 要采样的元素数量。
    :param time_horizon: [秒] 采样元素的时间范围。
    :param time_interval: [秒] 要采样的序列的时间间隔。
    :return: 能访问时序序列的采样索引。
    """
    if time_horizon <= 0.0 or time_interval <= 0.0 or time_horizon < time_interval:
        raise ValueError(
            f'Time horizon {time_horizon} must be greater or equal than target time interval {time_interval}'
            ' and both must be positive.'
        )

    # 计算步长和可采样的区间数
    num_intervals = int(time_horizon / time_interval) + 1
    step_size = num_intervals // num_samples

    assert step_size > 0, f"Cannot get {num_samples} samples in a {time_horizon}s horizon at {time_interval}s intervals"

    # 计算采样索引
    indices = list(range(step_size, num_intervals + 1, step_size))
    indices = indices[:num_samples]

    assert len(indices) == num_samples, f'Expected {num_samples} samples but only {len(indices)} were sampled'

    return indices
