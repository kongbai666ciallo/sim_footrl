from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.sim as sim_utils

from .my_task_env_cfg import MyTaskEnvCfg


class MyTaskEnv(DirectRLEnv):
    cfg: MyTaskEnvCfg

    def __init__(self, cfg: MyTaskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._external_torque_control: bool = False
        self.dof_idx = None
        self._last_actions = None
        self._last_q_targets = None
        self._last_dq = None
        # 指令缓冲: [vx, vy, yaw_rate] 以及重采样计数器
        self._commands = None  # 延后在拥有 num_envs 与 device 时初始化
        self._cmd_timer = None

    # --- Scene setup
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85))
        light_cfg.func("/World/Light", light_cfg)

    # --- Control
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 动作缓存（剪裁到 [-1,1]）
        if actions is None:
            if self.dof_idx is None:
                self.actions = None
            else:
                self.actions = torch.zeros((self.num_envs, len(self.dof_idx)), device=self.device)
        else:
            self.actions = torch.clamp(actions, -1.0, 1.0)
        # 初始化指令缓冲
        if self._commands is None:
            self._commands = torch.zeros((self.num_envs, 3), device=self.device)
            self._cmd_timer = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        # 指令重采样计时与触发
        self._cmd_timer += 1
        if self.cfg.cmd_resample_interval > 0:
            mask = self._cmd_timer >= self.cfg.cmd_resample_interval
            if mask.any():
                self._resample_commands(mask)

    def _apply_action(self) -> None:
        if self._external_torque_control:
            return
        # 将动作积分为目标关节位置（相对当前实际角度），并夹紧到软限位
        if self.dof_idx is None:
            return
        q = self.robot.data.joint_pos[:, self.dof_idx]
        q_targets = q + self.cfg.action_scale * self.actions
        # 夹紧到关节软限位
        limits = self.robot.data.soft_joint_pos_limits[:, self.dof_idx, :]
        q_targets = torch.clamp(q_targets, min=limits[..., 0], max=limits[..., 1])
        # 写入到控制接口（位置目标）
        try:
            self.robot.set_joint_position_target(q_targets, joint_ids=self.dof_idx)
        except Exception:
            # 退化路径：若接口不支持按 joint_ids，则直接全量设置
            self.robot.set_joint_position_target(q_targets)
        # 缓存
        self._last_q_targets = q_targets.clone()

    # --- Observations
    def _get_observations(self) -> dict:
        # 根部状态（在根坐标系）
        lin_vel_b = self.robot.data.root_com_lin_vel_b  # [N,3]
        ang_vel_b = self.robot.data.root_ang_vel_b      # [N,3]
        up_proj = self.robot.data.projected_gravity_b   # [N,3] 重力在机体坐标的投影，用于直立性
        # 关节状态（只取受控关节）
        q = self.robot.data.joint_pos[:, self.dof_idx]
        dq = self.robot.data.joint_vel[:, self.dof_idx]
        # 归一化：软限中心与范围
        limits = self.robot.data.soft_joint_pos_limits[:, self.dof_idx, :]
        center = 0.5 * (limits[..., 0] + limits[..., 1])
        span = torch.clamp(limits[..., 1] - limits[..., 0], min=1e-3)
        q_norm = (q - center) / (0.5 * span)
        # 上一步动作用于平滑（若首步则置零）
        if self._last_actions is None:
            if self.actions is None:
                # 初始化为零向量
                self._last_actions = torch.zeros((self.num_envs, len(self.dof_idx)), device=self.device)
            else:
                self._last_actions = torch.zeros_like(self.actions)
        # 指令缓存初始化
        if self._commands is None:
            self._commands = torch.zeros((self.num_envs, 3), device=self.device)
            self._cmd_timer = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        obs = [lin_vel_b, ang_vel_b, up_proj, q_norm, dq, self.actions, self._last_actions, self._commands]
        obs = torch.cat(obs, dim=-1)
        # 更新缓存
        self._last_actions = self.actions.clone()
        return {"policy": obs}

    # --- Rewards
    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        # 速度与姿态
        lin_vel_b = self.robot.data.root_com_lin_vel_b
        ang_vel_b = self.robot.data.root_ang_vel_b
        up_proj = self.robot.data.projected_gravity_b  # 朝向 (0,0,1) 越接近越直立

        # 指令（若未初始化则置零）: [vx_cmd, vy_cmd, yaw_cmd]
        if self._commands is None:
            self._commands = torch.zeros((self.num_envs, 3), device=self.device)
            self._cmd_timer = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        vx_cmd = self._commands[..., 0]
        vy_cmd = self._commands[..., 1]
        yaw_cmd = self._commands[..., 2]

        # 机体速度
        vx = lin_vel_b[..., 0]
        vy = lin_vel_b[..., 1]
        wz = ang_vel_b[..., 2]

        # 指数跟踪奖励
        r_track_vx_exp = torch.exp(-torch.square(vx - vx_cmd) / (2.0 * (cfg.track_vx_std ** 2)))
        r_track_vy_exp = torch.exp(-torch.square(vy - vy_cmd) / (2.0 * (cfg.track_vy_std ** 2)))
        r_track_yaw_exp = torch.exp(-torch.square(wz - yaw_cmd) / (2.0 * (cfg.track_yaw_std ** 2)))

        # 传统项（保持与原逻辑兼容）
        r_forward = -torch.square(vx - cfg.target_speed)
        upright = up_proj[..., 2]  # z轴上，重力朝下 => 接近 -1
        r_upright = -torch.square(upright + 1.0)
        r_lateral = -torch.square(vy)
        r_yaw_rate = -torch.square(wz)

        # 平整取向（重力投影在 x/y 平面越接近 0 越好）
        flat_xy = up_proj[..., :2]
        r_flat_orientation_l2 = -torch.sum(torch.square(flat_xy), dim=-1)

        # 动作与关节平滑
        if self._last_actions is None:
            self._last_actions = torch.zeros_like(self.actions)
        r_action_rate = -torch.mean(torch.square(self.actions - self._last_actions), dim=-1)
        dq = self.robot.data.joint_vel[:, self.dof_idx]
        if self._last_dq is None:
            self._last_dq = dq.clone()
        joint_acc = dq - self._last_dq
        r_joint_acc_l2 = -torch.sum(torch.square(joint_acc), dim=-1)
        r_joint_vel = -torch.mean(torch.square(dq), dim=-1)
        r_action = -torch.mean(torch.square(self.actions), dim=-1)

        rew = (
            # 指数跟踪
            cfg.w_track_vx_exp * r_track_vx_exp
            + cfg.w_track_vy_exp * r_track_vy_exp
            + cfg.w_track_yaw_exp * r_track_yaw_exp
            # 基础项
            + cfg.w_forward * r_forward
            + cfg.w_upright * r_upright
            + cfg.w_lateral * r_lateral
            + cfg.w_yaw_rate * r_yaw_rate
            # 姿态与平滑
            + cfg.w_flat_orientation_l2 * r_flat_orientation_l2
            + cfg.w_action_rate * r_action_rate
            + cfg.w_joint_vel * r_joint_vel
            + cfg.w_action * r_action
            + cfg.w_joint_acc_l2 * r_joint_acc_l2
        )
        # 更新缓存
        self._last_dq = dq.detach()
        # RSL-RL 期望 shape 为 [N] 的一维张量
        return rew

    # --- Dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        reset = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # 根部高度（COM）太低 -> 摔倒
        z = self.robot.data.root_pos_w[..., 2]
        fall = z < cfg.min_base_height
        # 直立性（通过 projected_gravity_b 的 z 分量近似）
        tilt_bad = torch.acos(torch.clamp(-self.robot.data.projected_gravity_b[..., 2], -1.0, 1.0)) > cfg.max_tilt_rad
        reset = torch.logical_or(fall, tilt_bad)
        # 宽限期：前 init_grace_steps 步不因为跌倒/倾斜结束
        grace_mask = self.episode_length_buf < self.cfg.init_grace_steps
        reset[grace_mask] = False
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return reset, time_out

    # --- Reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if self.dof_idx is None:
            self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_name)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        # 放置到对应环境原点
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        # 设置初始高度
        default_root_state[:, 2] = self.cfg.init_base_height
        # 设置朝向为正立（四元数 w=1,x=y=z=0 已满足）
        default_root_state[:, 3:7] = torch.tensor([0, 0, 0, 1], device=self.device).repeat(len(env_ids), 1)
        self.robot.write_root_state_to_sim(default_root_state, env_ids)
        # 缓存按 env_ids 清理/初始化
        if self._last_actions is None:
            self._last_actions = torch.zeros((self.num_envs, len(self.dof_idx)), device=self.device)
        else:
            self._last_actions[env_ids] = 0.0
        if self._last_q_targets is None:
            self._last_q_targets = torch.zeros((self.num_envs, len(self.dof_idx)), device=self.device)
        self._last_q_targets[env_ids] = self.robot.data.joint_pos[env_ids][:, self.dof_idx].clone()
        if self._last_dq is None:
            self._last_dq = torch.zeros((self.num_envs, len(self.dof_idx)), device=self.device)
        else:
            self._last_dq[env_ids] = 0.0
        # 指令缓存
        if self._commands is None:
            self._commands = torch.zeros((self.num_envs, 3), device=self.device)
            self._cmd_timer = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._commands[env_ids] = 0.0
        self._cmd_timer[env_ids] = 0
        # 初始关节站立姿态（若存在）
        pose_cfg = self.cfg.init_stand_pose
        if pose_cfg and self.cfg.enable_init_stand_pose:
            joint_names_all = self.robot.data.joint_names
            name_to_index = {n: i for i, n in enumerate(joint_names_all)}
            # 全量关节 pos/vel（形状 [num_envs, num_joints]）
            full_q = self.robot.data.joint_pos.clone()
            full_dq = self.robot.data.joint_vel.clone()
            # 修改指定 env_ids 的对应关节
            for name, val in pose_cfg.items():
                idx = name_to_index.get(name)
                if idx is None:
                    continue
                full_q[env_ids, idx] = val
            # 直接写全量，不传 env_ids（内部会广播到所有 env），随后再写 root_state 已完成
            # 若仅写 env_ids 可尝试不传 joint_ids 以避免内部 advanced indexing 触发形状冲突
            self.robot.write_joint_state_to_sim(full_q, full_dq)
            # 同步受控关节的目标缓存（只对复位的 env）
            self._last_q_targets[env_ids] = full_q[env_ids][:, self.dof_idx]
            if self.cfg.debug_init:
                print("[DEBUG init] env_ids=", env_ids,
                      "root_z=", self.robot.data.root_pos_w[env_ids, 2],
                      "first_controlled_q=", full_q[env_ids][:, self.dof_idx][0, :5])

    # 外部切换控制模式
    def enable_external_torque_control(self, enabled: bool = True):
        self._external_torque_control = bool(enabled)
        print(f"[INFO] MyTaskEnv external torque control = {self._external_torque_control}")

    @torch.no_grad()
    def _resample_commands(self, mask: torch.Tensor):
        # 从配置范围内重采样 [vx, vy, yaw_rate]
        low = torch.tensor([
            self.cfg.cmd_lin_vel_x_range[0],
            self.cfg.cmd_lin_vel_y_range[0],
            self.cfg.cmd_ang_vel_z_range[0],
        ], device=self.device)
        high = torch.tensor([
            self.cfg.cmd_lin_vel_x_range[1],
            self.cfg.cmd_lin_vel_y_range[1],
            self.cfg.cmd_ang_vel_z_range[1],
        ], device=self.device)
        num = int(mask.sum().item())
        if num <= 0:
            return
        rand = torch.rand((num, 3), device=self.device)
        cmds = low + (high - low) * rand
        self._commands[mask] = cmds
        self._cmd_timer[mask] = 0
