# footrl

`footrl` 是一个基于 Isaac Sim / Isaac Lab 的下肢（足部/步态）强化学习实验包示例。它从原始仓库中拆分，只包含与步态训练相关的脚本与扩展别名，不包含主工程的其它资产与任务。

## 目录结构
```
footrl/
  README.md              # 本说明
  LICENSE                # BSD-3 License
  source/sim_1/          # Isaac Lab 扩展别名 (包含 tasks 注册与 env cfg)
  scripts/               # 训练 / 推理 / RL 框架脚本
    rsl_rl/              # rsl-rl 专用入口 (train/play)
```

## 功能概览
- 自定义任务 `Template-MyTask-Direct-v0`，直接环境（DirectRLEnv 接口）
- RSL-RL 训练脚本：`scripts/rsl_rl/train.py`
- RSL-RL 推理脚本：`scripts/rsl_rl/play.py`
- 独立 agents 配置（`rsl_rl_ppo_cfg.py`）位于 `footrl/source/sim_1/footrl/tasks/direct/my_task/agents/`

## 运行前提
1. 已安装并能运行 Isaac Sim + Isaac Lab (需要 Omniverse Kit 环境)。
2. Python 环境中可访问扩展依赖包：`isaaclab`, `isaaclab_tasks`, `isaaclab_rl`, `rsl_rl_lib`。

## 安装方式
进入你的isaaclab创建的conda环境
```bash
conda activate isaaclab
```
进入目录安装
```bash
cd footrl/source/sim1
pip install -e .
```
## 启动训练
```bash
python /absolute/path/to/footrl/scripts/rsl_rl/train.py \
  --task Template-MyTask-Direct-v0 \
  --num_envs 16 \
  --max_iterations 1000 \
  --device cuda:0
```

## 推理/回放
```bash
python /absolute/path/to/footrl/scripts/rsl_rl/play.py \
  --task Template-MyTask-Direct-v0 \
  --checkpoint /path/to/logs/rsl_rl/my_task_direct/<run>/checkpoints/model_*.pt
```

