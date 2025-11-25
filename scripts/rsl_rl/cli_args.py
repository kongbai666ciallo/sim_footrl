"""RSL-RL 命令行参数与配置更新模块。

复制自上级 `scripts/rsl_rl/cli_args.py`，保持接口兼容：
 - add_rsl_rl_args(parser)
 - parse_rsl_rl_cfg(task_name, args_cli)
 - update_rsl_rl_cfg(agent_cfg, args_cli)

这样可避免在运行 `footrl/scripts/rsl_rl/train.py` 与 `play.py` 时出现 ModuleNotFoundError。
后续如果需要消除重复，可在这里改为从上级模块动态导入。现在先保证独立可用。
"""

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """向解析器添加 RSL-RL 相关参数。"""
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> "RslRlBaseRunnerCfg":
    """根据任务名与 CLI 参数解析默认 RSL-RL 配置，并应用更新。"""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    rslrl_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: "RslRlBaseRunnerCfg", args_cli: argparse.Namespace):
    """根据 CLI 参数更新 RSL-RL 配置对象。"""
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if getattr(args_cli, "resume", None) is not None:
        agent_cfg.resume = args_cli.resume
    if getattr(args_cli, "load_run", None) is not None:
        agent_cfg.load_run = args_cli.load_run
    if getattr(args_cli, "checkpoint", None) is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if getattr(args_cli, "run_name", None) is not None:
        agent_cfg.run_name = args_cli.run_name
    if getattr(args_cli, "logger", None) is not None:
        agent_cfg.logger = args_cli.logger

    if agent_cfg.logger in {"wandb", "neptune"} and getattr(args_cli, "log_project_name", None):
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
