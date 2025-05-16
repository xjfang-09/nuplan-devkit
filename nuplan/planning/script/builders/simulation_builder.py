import logging
import os
from typing import List, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.common.utils.distributed_scenario_filter import DistributedMode, DistributedScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.script.builders.metric_builder import build_metrics_engines
from nuplan.planning.script.builders.observation_builder import build_observations
from nuplan.planning.script.builders.planner_builder import build_planners
from nuplan.planning.script.builders.utils.utils_type import is_target_type
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.simulation_time_controller.abstract_simulation_time_controller import (
    AbstractSimulationTimeController,
)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool

logger = logging.getLogger(__name__)


def build_simulations(
    cfg: DictConfig,
    worker: WorkerPool,
    callbacks: List[AbstractCallback],
    callbacks_worker: Optional[WorkerPool] = None,
    pre_built_planners: Optional[List[AbstractPlanner]] = None,
) -> List[SimulationRunner]:
    """
    构建模拟。
    :param cfg: DictConfig. 用于运行实验的配置。
    :param callbacks: 模拟的回调函数。
    :param worker: 用于任务执行的工作池。
    :param callbacks_worker: 用于模拟回调的工作池。
    :param pre_built_planners: 预构建的规划器列表，用于运行模拟。
    :return: 包含挑战名称的模拟引擎字典。
    """
    logger.info('正在构建模拟...')

    # 创建 Simulation 对象容器
    simulations = list()

    # 获取场景
    logger.info('提取场景...')

    # 仅允许使用 NuPlanScenarioBuilder 进行模拟，除非环境变量 NUPLAN_SIMULATION_ALLOW_ANY_BUILDER 设置为非零值。
    if not int(os.environ.get("NUPLAN_SIMULATION_ALLOW_ANY_BUILDER", "0")) and not is_target_type(
        cfg.scenario_builder, NuPlanScenarioBuilder
    ):
        raise ValueError(f"模拟框架仅支持 NuPlanScenarioBuilder。当前为 {cfg.scenario_builder}")

    scenario_filter = DistributedScenarioFilter(
        cfg=cfg,
        worker=worker,
        node_rank=int(os.environ.get("NODE_RANK", 0)),
        num_nodes=int(os.environ.get("NUM_NODES", 1)),
        synchronization_path=cfg.output_dir,
        timeout_seconds=cfg.distributed_timeout_seconds,
        distributed_mode=DistributedMode[cfg.distributed_mode],
    )
    scenarios = scenario_filter.get_scenarios()

    metric_engines_map = {}
    if cfg.run_metric:
        logger.info('正在构建指标引擎...')
        metric_engines_map = build_metrics_engines(cfg=cfg, scenarios=scenarios)
        logger.info('指标引擎构建完成')
    else:
        logger.info('指标引擎已禁用')

    logger.info('从 %d 个场景构建模拟...', len(scenarios))

    # 构建指标元数据文件
    for scenario in scenarios:

        # 构建规划器
        if pre_built_planners is None:
            if 'planner' not in cfg.keys():
                raise KeyError('配置中未指定规划器。请使用 "planner" 字段指定规划器。')

            planners = build_planners(cfg.planner, scenario)
        else:
            planners = pre_built_planners

        for planner in planners:
            # 自车控制器
            ego_controller: AbstractEgoController = instantiate(cfg.ego_controller, scenario=scenario)

            # 模拟管理器
            simulation_time_controller: AbstractSimulationTimeController = instantiate(
                cfg.simulation_time_controller, scenario=scenario
            )

            # 感知
            observations: AbstractObservation = build_observations(cfg.observation, scenario=scenario)

            # 指标引擎
            metric_engine = metric_engines_map.get(scenario.scenario_type, None)
            if metric_engine is not None:
                stateful_callbacks = [MetricCallback(metric_engine=metric_engine, worker_pool=callbacks_worker)]
            else:
                stateful_callbacks = []

            if "simulation_log_callback" in cfg.callback:
                stateful_callbacks.append(
                    instantiate(cfg.callback["simulation_log_callback"], worker_pool=callbacks_worker)
                )

            # 构建模拟和管理器
            simulation_setup = SimulationSetup(
                time_controller=simulation_time_controller,
                observations=observations,
                ego_controller=ego_controller,
                scenario=scenario,
            )

            simulation = Simulation(
                simulation_setup=simulation_setup,
                callback=MultiCallback(callbacks + stateful_callbacks),
                simulation_history_buffer_duration=cfg.simulation_history_buffer_duration,
            )
            simulations.append(SimulationRunner(simulation, planner))

    logger.info('模拟构建完成！')
    return simulations