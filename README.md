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
推荐在 Isaac Sim 的 Python 下进行可编辑安装：
```bash
/path/to/isaac-sim/python.sh -m pip install -e /absolute/path/to/footrl/source/sim_1
```
> 如果你已经在主仓库中，该路径可能类似：`/home/USER/Download/project/sim_1/footrl/source/sim_1`。

安装完成后可以验证扩展加载：
```python
python -c "import footrl.tasks; print('footrl tasks imported OK')"
```

## 启动训练
```bash
/path/to/isaac-sim/python.sh /absolute/path/to/footrl/scripts/rsl_rl/train.py \
  --task Template-MyTask-Direct-v0 \
  --num_envs 16 \
  --max_iterations 1000 \
  --device cuda:0
```

## 推理/回放
```bash
/path/to/isaac-sim/python.sh /absolute/path/to/footrl/scripts/rsl_rl/play.py \
  --task Template-MyTask-Direct-v0 \
  --checkpoint /path/to/logs/rsl_rl/my_task_direct/<run>/checkpoints/model_*.pt
```

## 常见问题
| 问题 | 可能原因 | 解决办法 |
|------|----------|----------|
| ModuleNotFoundError: isaaclab.* | 未在 Isaac Sim 提供的 Python 下运行 | 使用 Isaac Sim `python.sh` 启动脚本 |
| 无法导入 footrl.tasks | 未执行 pip 安装或路径错误 | 确认 `pip install -e source/sim_1` 成功 |
| 训练奖励很低 | 初始策略为随机，观察归一化或奖励权重需调节 | 调整 `rsl_rl_ppo_cfg.py` 中网络结构或 gamma/entropy |

## 后续改进建议
- 增加多步态命令采样（速度 / 方向随机化）。
- 添加多 Agent 扩展示例（DirectMARL）。
- 引入 wandb / tensorboard 记录（启用 `--logger` 参数）。
- 编写更丰富的奖励项与终止条件。

## License
项目沿用 BSD 3-Clause (见 LICENSE)。

## 致谢
基于 Isaac Lab 项目结构和 RSL-RL 框架示例进行裁剪整合。
