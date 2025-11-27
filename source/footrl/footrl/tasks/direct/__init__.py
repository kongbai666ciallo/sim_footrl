# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym  # noqa: F401

# 触发本目录下子任务的注册（direct.hi_test、direct.sim_1 等）
try:  # 避免在构建/检查时硬错误
	from . import hi_test as _hi_test  # noqa: F401
except Exception:  # pragma: no cover
	pass

try:  # 同时触发内置 sim_1 的注册（如需要）
	from . import sim_1 as _sim1  # noqa: F401
except Exception:  # pragma: no cover
	pass

# 明确导入自定义 my_task，确保注册
try:
    from . import my_task as _my_task  # noqa: F401
except Exception:  # pragma: no cover
    pass
