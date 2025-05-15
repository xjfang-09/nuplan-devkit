from __future__ import annotations

import abc
from enum import Enum
from typing import List, Type

from nuplan.common.maps.abstract_map_factory import AbstractMapFactory
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool


class RepartitionStrategy(Enum):
    """分布式场景缓存时使用的重新分区策略。"""

    REPARTITION_FILE_DISK = 1  # 从文件加载后进行重分布以实现负载均衡
    INLINE = 2  # 每个 worker 上构建全部场景，然后均匀分发


class AbstractScenarioBuilder(abc.ABC):
    """通用场景构造器接口。"""

    @classmethod
    @abc.abstractmethod
    def get_scenario_type(cls) -> Type[AbstractScenario]:
        """获取该构建器所构造的场景类型。"""
        pass

    @abc.abstractmethod
    def get_scenarios(self, scenario_filter: ScenarioFilter, worker: WorkerPool) -> List[AbstractScenario]:
        """
        从数据库中提取过滤后的场景。
        :param scenario_filter: 包含场景过滤指令的结构体。
        :param worker: 并行处理用的线程池。
        :return: 场景列表。
        """
        pass

    @abc.abstractmethod
    def get_map_factory(self) -> AbstractMapFactory:
        """
        获取地图工厂实例。
        """
        pass

    @property
    def repartition_strategy(self) -> RepartitionStrategy:
        """
        获取在分布式设置中用于缓存的重新分区策略。
        """
        pass