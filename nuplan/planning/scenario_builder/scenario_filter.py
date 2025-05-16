from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union


@dataclass(frozen=True)
class ScenarioFilter:
    """
    用于从数据库中构建训练/仿真场景的过滤器集合。
    """

    # 要包含的场景类型列表：
    scenario_types: Optional[List[str]]
    # 要包含的场景列表，形式为 (log_name, token)：
    scenario_tokens: Optional[List[Sequence[str]]]

    # 按日志名称过滤场景：
    log_names: Optional[List[str]]
    # 按地图名称过滤场景：
    map_names: Optional[List[str]]

    # 每种类型的场景数量：
    num_scenarios_per_type: Optional[int]
    # 限制总场景数量（float = 比例，int = 数量）：
    limit_total_scenarios: Optional[Union[int, float]]
    # 场景初始激光雷达时间戳之间的时间间隔阈值（秒）：
    timestamp_threshold_s: Optional[float]

    # 自车中心覆盖的总距离（逐帧计算）的最小阈值（米），用于保留场景：
    ego_displacement_minimum_m: Optional[float]

    # 是否将多样本场景扩展为多个单样本场景：
    expand_scenarios: bool
    # 是否移除任务目标无效的场景：
    remove_invalid_goals: bool
    # 是否对场景进行随机排序：
    shuffle: bool

    # 自车速度必须超过的排除阈值（米/秒），用于保留场景：
    ego_start_speed_threshold: Optional[float] = None
    # 自车速度必须低于的包含阈值（米/秒），用于保留场景：
    ego_stop_speed_threshold: Optional[float] = None
    # 两个时间点之间的速度变化低于该值时，将被视为噪声忽略：
    speed_noise_tolerance: Optional[float] = None

    # 指向包含 Nuplan 数据库中 lidarpc token 集的 json 文件的路径，我们希望场景包含这些 token：
    token_set_path: Optional[Path] = None

    # 阈值范围为 [0, 1]。
    # 如果为 1，场景必须仅包含 token_set_path（见上文）中的 lidarpc token，才能通过过滤器。
    # 如果在 [0, 1) 范围内，场景仅在其 lidarpc token 中包含的比例严格大于以下阈值时才能通过：
    fraction_in_token_set_threshold: Optional[float] = None

    # 自车周围的半径，用于检查是否存在在路线上行驶的车道段
    # 使用 VectorMap 收集车道段和路线状态
    # 用于过滤掉没有路线的场景
    ego_route_radius: Optional[float] = None

    def __post_init__(self) -> None:
        """清理类属性。"""
        if self.num_scenarios_per_type is not None:
            assert 0 < self.num_scenarios_per_type, "num_scenarios_per_type 应为正整数"

        if isinstance(self.limit_total_scenarios, float):
            assert 0.0 < self.limit_total_scenarios <= 1.0, "当 limit_total_scenarios 为 float 时，其值应在 (0, 1] 范围内"
        elif isinstance(self.limit_total_scenarios, int):
            assert 0 < self.limit_total_scenarios, "当 limit_total_scenarios 为整数时，其值应为正数"
