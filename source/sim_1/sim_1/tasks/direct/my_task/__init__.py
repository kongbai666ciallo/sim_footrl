"""MyTask: 任务注册入口。

将本包导入后自动向 Gym 注册环境。
"""

from __future__ import annotations

import gymnasium as gym

# 复用现有 sim_1 扩展中的 agents 配置，使用绝对导入避免在别名/裁剪的副本中找不到相对路径
try:
    # 优先使用 footrl 自己的 agents 包（新命名空间）
    from footrl.tasks.direct.my_task import agents as _base_agents  # type: ignore
except ImportError:
    # 回退到原扩展下的 sim_1 agents
    from sim_1.tasks.direct.sim_1 import agents as _base_agents  # type: ignore


ENV_ID = "Template-MyTask-Direct-v0"


def _ensure_registered():
    """在重复 import 时避免重复注册。"""
    try:
        gym.spec(ENV_ID)
        return
    except Exception:
        pass

    gym.register(
        id=ENV_ID,
        entry_point=f"{__name__}.my_task_env:MyTaskEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.my_task_env_cfg:MyTaskEnvCfg",
            "rl_games_cfg_entry_point": f"{_base_agents.__name__}:rl_games_ppo_cfg.yaml",
            "rsl_rl_cfg_entry_point": f"{_base_agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
            "skrl_cfg_entry_point": f"{_base_agents.__name__}:skrl_ppo_cfg.yaml",
            "sb3_cfg_entry_point": f"{_base_agents.__name__}:sb3_ppo_cfg.yaml",
        },
    )


_ensure_registered()
