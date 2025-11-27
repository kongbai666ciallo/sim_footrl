from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# 仅使用重命名后的包路径，避免引用工作区外的旧包名
from footrl.robot import HI_CFG  # type: ignore


@configclass
class MyTaskEnvCfg(DirectRLEnvCfg):
    """HI 机器人步态训练任务配置。

    关键项：
    - dof_name: 控制的关节名称列表（默认覆盖下肢与腰部，可按需扩展/裁剪）。
    - action/obs 维度：由 env 根据 dof 自动推断；此处保留占位。
    - 任务参数：目标前进速度、动作尺度、奖励权重与结束条件。
    """

    # 控制频率：以物理步长 dt=1/120，decimation 步执行一次控制
    decimation = 4  # 控制频率约 30 Hz
    episode_length_s = 20.0

    # 以步态为主：腰部 + 双腿（肩臂默认不控制）
    dof_name = [
        "waist_joint",
        # 左腿
        "l_hip_pitch_joint", "l_hip_roll_joint", "l_thigh_joint", "l_calf_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
        # 右腿
        "r_hip_pitch_joint", "r_hip_roll_joint", "r_thigh_joint", "r_calf_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint",
    ]

    # 由环境在构造时推断；这里仅占位
    action_space = len(dof_name)
    observation_space = 64
    state_space = 0

    # 物理与场景
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    robot_cfg: ArticulationCfg = HI_CFG.replace(prim_path="/my_task/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=3.0, replicate_physics=True)

    # 动作/目标
    action_scale: float = 0.25  # 每步允许的目标关节增量（rad）
    target_speed: float = 0.8   # 期望前进速度 m/s

    # 奖励权重
    w_forward: float = 2.0      # 靠近目标速度
    w_upright: float = 1.0      # 保持躯干直立
    w_lateral: float = 0.2      # 侧向速度惩罚
    w_yaw_rate: float = 0.1     # 偏航角速度惩罚
    w_action_rate: float = 0.01 # 动作变化平滑
    w_joint_vel: float = 0.005  # 关节速度惩罚
    w_action: float = 0.002     # 动作幅值惩罚

    # 指令跟踪（对标 LeggedLab 风格）
    cmd_lin_vel_x_range: tuple[float, float] = (-1.0, 1.5)
    cmd_lin_vel_y_range: tuple[float, float] = (-0.5, 0.5)
    cmd_ang_vel_z_range: tuple[float, float] = (-1.0, 1.0)
    cmd_resample_interval: int = 50  # 每隔多少个 env 步重采样一次指令
    # 指数跟踪的标准差（越小越严格）
    track_vx_std: float = 0.25
    track_vy_std: float = 0.45
    track_yaw_std: float = 0.50
    # 新增奖励项权重（指数跟踪 + 平整 + 加速度与动作平滑）
    w_track_vx_exp: float = 1.0
    w_track_vy_exp: float = 2.0
    w_track_yaw_exp: float = 2.0
    w_flat_orientation_l2: float = 2.0
    w_joint_acc_l2: float = 1e-4

    # 结束条件
    min_base_height: float = 0.35
    max_tilt_rad: float = 0.6  # 俯仰/横滚过大

    # 初始化参数
    init_base_height: float = 0.6           # 复位时根部高度 (m)
    init_grace_steps: int = 30              # 前若干步不因跌倒/倾斜结束（防止刚起步抖动重置）
    # 简易站立姿态（若关节存在则写入），根据双腿轻微屈膝以稳定：
    # 数值可后续基于真实模型调节；不存在的关节将被忽略。
    init_stand_pose: dict[str, float] | None = {
        "l_hip_pitch_joint": 0.0,
        "l_hip_roll_joint": 0.0,
        "l_thigh_joint": 0.0,
        "l_calf_joint": 0.0,
        "l_ankle_pitch_joint": 0.0,
        "l_ankle_roll_joint": 0.0,
        "r_hip_pitch_joint": 0.0,
        "r_hip_roll_joint": 0.0,
        "r_thigh_joint": 0.0,
        "r_calf_joint": 0.0,
        "r_ankle_pitch_joint": 0.0,
        "r_ankle_roll_joint": 0.0,
        "waist_joint": 0.0,
    }
    enable_init_stand_pose: bool = True   # 是否在 reset 时写入站立姿态
    debug_init: bool = False              # 是否打印调试信息（root高度/关节前几个值）
