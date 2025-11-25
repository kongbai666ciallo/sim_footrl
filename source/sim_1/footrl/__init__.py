"""footrl 包 (别名层):

为现有 sim_1 扩展提供新的导入名 `footrl`，不移动源码，通过动态转发子模块实现：
    - from footrl import *            -> 转发到 sim_1.*
    - from footrl.robot import HI_CFG -> 等价于 from sim_1.robot import HI_CFG

这样外部代码可以统一使用 `footrl` 作为包名。
"""
from __future__ import annotations
import importlib
import sys

# 直接导入 sim_1 顶层，以复用其初始化副作用（环境注册等）
_sim1 = importlib.import_module("sim_1")

# 将常用子模块映射到当前命名空间，支持 from footrl.robot import ...
_SUBMODULES = [
    "assets",
    "data",
    "envs",
    "robot",
    "tasks",
    "ui_extension_example",
]
for _name in _SUBMODULES:
    try:
        sys.modules[__name__ + "." + _name] = importlib.import_module("sim_1." + _name)
    except Exception:  # 子模块可能不存在，忽略即可
        pass

# 通配导出：尽量与 sim_1 保持一致
try:
    from sim_1 import *  # noqa: F401,F403
except Exception:
    pass
