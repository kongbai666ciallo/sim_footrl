# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the operational space controller (OSC) with the simulator.

The OSC controller can be configured in different modes. It uses the dynamical quantities such as Jacobians and
mass matricescomputed by PhysX.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_osc.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the operational space controller.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# 正弦抖动参数：幅度(度)与频率(Hz)
parser.add_argument("--sin-amp-deg", type=float, default=0.0, help="Finger oscillation amplitude in degrees (around straight pose).")
parser.add_argument("--sin-freq-hz", type=float, default=0.0, help="Finger oscillation frequency in Hz.")
parser.add_argument("--no-gravity", action="store_true", help="Disable gravity for the hand (robot-level).")
parser.add_argument("--fixed-base", action="store_true", help="Fix the robot base to the world (default: free base).")
parser.add_argument("--mode", type=str, choices=["vel", "osc", "torque"], default="torque", help="Control mode: 'torque' uses joint-space PD torques; 'vel' uses velocity-PD; 'osc' reserved for OSC.")
parser.add_argument("--ros2-cmd-topic", type=str, default="/joint_cmd_array", help="ROS2 订阅的命令话题，消息内容为字符串 '[[name, angle], ...]' ")
parser.add_argument("--ros2-cmd-unit", type=str, choices=["rad", "deg"], default="rad", help="ROS2 命令角度单位")
parser.add_argument("--ros2-joint-state-topic", type=str, default="/joint_states", help="ROS2 JointState 发布的话题名")
parser.add_argument("--ros2-joint-state-rate", type=float, default=1500.0, help="JointState 发布频率 (Hz)，设为 0 可关闭发布")
parser.add_argument("--ros2-joint-state-qos", type=str, choices=["reliable", "best_effort", "sensor"], default="reliable", help="JointState 发布 QoS 模式：可靠(reliable)/尽力而为(best_effort)/传感器(sensor)")
parser.add_argument("--ros2-joint-state-scope", type=str, choices=["all", "selected"], default="all", help="JointState 发布的关节范围：全部(all)/仅选中的上肢(selected)")
parser.add_argument("--ros2-joint-state-threaded", action="store_true", help="强制使用线程发布（绕过 rcl timer），在高频下更稳")
parser.add_argument("--torque-kp", type=float, default=5.0, help="力控关节 Kp (Nm/rad)")
parser.add_argument("--torque-kd", type=float, default=1, help="力控关节 Kd (Nm/(rad/s))")
parser.add_argument("--effort-limit", type=float, default=50000.0, help="力矩限幅 |tau|<=limit")
parser.add_argument("--zero-drive-in-torque", action="store_true", help="力控模式下将关节驱动 stiffness/damping 清零以避免与力控叠加")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
# ...existing code...
import os, sys, threading
# 将 <repo>/source 加入 sys.path，便于绝对导入 sim_1 包
_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
_SRC_DIR = os.path.join(_REPO_ROOT, "source")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from sim_1.robot import HI_CFG  # noqa: E402
# ...existing code...
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

# --- Utilities: fix base link to world using PhysX FixedJoint ---
def _get_stage_via_usd():
    try:
        import omni.usd
        return omni.usd.get_context().get_stage()
    except Exception:
        return None

def _find_robot_root_prim(stage, explicit_path_candidates=None):
    from pxr import Usd
    if explicit_path_candidates is None:
        explicit_path_candidates = [
            "/World/envs/env_0/Robot",
            "/World/Robot",
        ]
    for p in explicit_path_candidates:
        prim = stage.GetPrimAtPath(p)
        if prim and prim.IsValid():
            return prim
    # fallback scan under /World
    world = stage.GetPrimAtPath("/World")
    for prim in Usd.PrimRange(world):
        if prim and prim.IsValid() and prim.GetPath().pathString.endswith("/Robot"):
            return prim
    return None

def fix_base_to_world(base_name_candidates=(
    "base_link", "waist_link", "pelvis", "pelvis_link", "torso", "torso_link", "base", "Root", "root"
)) -> bool:
    """Create a PhysX FixedJoint to weld the robot base link to world.
    Returns True if created, False otherwise.
    """
    try:
        from pxr import Usd, UsdPhysics, Sdf, Gf, UsdGeom
        stage = _get_stage_via_usd()
        if stage is None:
            print("[warn] 获取 USD Stage 失败，无法固定基座。")
            return False
        robot_root = _find_robot_root_prim(stage)
        if robot_root is None:
            print("[warn] 未找到 Robot 根 prim，无法固定基座。")
            return False
        # locate base link prim
        base_prim = None
        base_name = None
        root_path = robot_root.GetPath().pathString
        for name in base_name_candidates:
            p = stage.GetPrimAtPath(f"{root_path}/{name}")
            if p and p.IsValid():
                base_prim = p
                base_name = name
                break
        if base_prim is None:
            print("[warn] 未在 Robot 下找到基座链接，放弃固定。")
            return False
        # create or redefine a FixedJoint to world
        joint_path = "/World/HangJoint"
        joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(joint_path))
        joint.CreateBody0Rel().SetTargets([Sdf.Path(base_prim.GetPath())])
        # Do not set Body1Rel => fixed to world
        # place world anchor at base link current world pose to avoid initial pull
        try:
            xcache = UsdGeom.XformCache()
            m = xcache.GetLocalToWorldTransform(base_prim)
            # Extract translation
            if hasattr(m, "ExtractTranslation"):
                pos = m.ExtractTranslation()
                pos = (float(pos[0]), float(pos[1]), float(pos[2]))
            else:
                pos = (float(m[3][0]), float(m[3][1]), float(m[3][2]))
            if hasattr(joint, 'CreateLocalPos1Attr'):
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*pos))
            if hasattr(joint, 'CreateLocalRot1Attr'):
                joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        except Exception:
            pass
        print(f"[INFO] 已将 {base_name} 通过 FixedJoint 固定到世界 ({joint_path})")
        return True
    except Exception as e:
        print(f"[ERROR] 固定 base_link 失败: {e}")
        return False

def _prepare_hi_robot_cfg(base_cfg, no_gravity: bool, fixed_base: bool):
    """安全调整 HI 资产配置：
    - 若 spawn.rigid_props 存在且 no_gravity=True，则置 disable_gravity=True。
    - HI 的 UsdFileCfg 通常没有 fix_base 字段，这里仅打印提示。
    """
    spawn = base_cfg.spawn
    _rp = getattr(spawn, "rigid_props", None)
    if no_gravity and _rp is not None:
        try:
            spawn = spawn.replace(rigid_props=_rp.replace(disable_gravity=True))
        except Exception:
            print("[warn] rigid_props 不支持 replace，忽略 --no-gravity")
    if fixed_base:
        print("[warn] HI 资产未提供 fix_base 字段；如需固定基座，请在场景中添加 Weld/Fix Joint 到世界。")
    return base_cfg.replace(spawn=spawn)

##
# Pre-defined configs
##
# from ..source.sim_hand.sim_hand.robot.hand import L10_CFG  # isort:skip


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a simple scene with a tilted wall."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Tilted wall
    tilted_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TiltedWall",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 1.5, 0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            activate_contact_sensors=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.6 + 0.085, 0.0, 0.3), rot=(0.9238795325, 0.0, -0.3826834324, 0.0)
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/TiltedWall",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
    )

    # 安全准备 HI 机器人配置（仅在可用字段上做最小改动）
    robot = _prepare_hi_robot_cfg(
        HI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        args_cli.no_gravity,
        args_cli.fixed_base,
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop.

    Args:
        sim: (SimulationContext) Simulation context.
        scene: (InteractiveScene) Interactive scene.
    """

    # Extract scene entities for readability.
    robot = scene["robot"]
    contact_forces = scene["contact_forces"]

    # 固定基座：默认将 base_link 通过 FixedJoint 固定到世界
    fix_base_to_world()

    # Obtain indices for the end-effector and arm joints
    ee_candidates = ["base_link", "palm", "hand"]
    ee_frame_idx = None
    for pat in ee_candidates:
        try:
            ids = robot.find_bodies(pat)[0]
            if len(ids) > 0:
                ee_frame_idx = int(ids[0])
                break
        except Exception:
            pass
    if ee_frame_idx is None:
        ee_frame_idx = 0
        print(f"[warn] 未找到 {ee_candidates}，回退使用 body index 0。部分名称样例：{robot.body_names[:10]}")

    # 2) 关节集合：按名称关键词选择上肢关节（肩/上臂/肘/腕）
    name_filters = ("shoulder", "upper_arm", "elbow", "wrist")
    all_joint_names = robot.joint_names
    arm_joint_ids = [i for i, n in enumerate(all_joint_names) if any(k in n for k in name_filters)]
    if arm_joint_ids is None or len(arm_joint_ids) == 0:
        # 若未命中，退化为所有关节
        arm_joint_ids = list(range(robot.num_joints))
        print(f"[warn] 未匹配到上肢关节，退化为全部关节。样例关节名：{all_joint_names[:20]}")
    else:
        sel_names_preview = [all_joint_names[j] for j in arm_joint_ids[:10]]
        print(f"[info] 命中上肢关节数={len(arm_joint_ids)}，例：{sel_names_preview}")

    # 统一为张量索引/列表
    arm_joint_ids = torch.as_tensor(arm_joint_ids, device=sim.device, dtype=torch.long)
    arm_joint_ids_list = arm_joint_ids.tolist()
    dof_sel = int(arm_joint_ids.numel())

    # 简化：将选中关节全作为可抖动集合
    selected_joint_names = [robot.joint_names[j] for j in arm_joint_ids_list]
    base_idxs = []
    flex_idxs = list(range(dof_sel))

    print(f"[debug] ee_frame_idx={ee_frame_idx}, num_selected_joints={dof_sel}")

    # JointState 发布范围：全部或仅选中关节
    if args_cli.ros2_joint_state_scope == "selected":
        js_idx_list = arm_joint_ids_list
        js_names = [robot.joint_names[j] for j in js_idx_list]
    else:
        js_idx_list = list(range(robot.num_joints))
        js_names = list(robot.joint_names)

    # === 启用 OSC：仅做 nullspace 关节位姿控制（禁用任务空间轴），并开启重力补偿 ===
    # osc_cfg = OperationalSpaceControllerCfg(
    #     target_types=["pose_abs"],               # 保留占位类型
    #     impedance_mode="variable_kp",
    #     inertial_dynamics_decoupling=False,       # 关闭惯量解耦，避免需要质量矩阵逆
    #     partial_inertial_dynamics_decoupling=False,
    #     gravity_compensation=True,                # 开启重力补偿
    #     motion_damping_ratio_task=1.0,
    #     contact_wrench_stiffness_task=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     motion_control_axes_task=[0, 0, 0, 0, 0, 0],  # 禁用任务空间位姿轴
    #     contact_wrench_control_axes_task=[0, 0, 0, 0, 0, 0],
    #     nullspace_control="position",            # 在零空间做关节位置控制
    # )
    # osc = OperationalSpaceController(osc_cfg, num_envs=scene.num_envs, device=sim.device)

    # # Markers
    # frame_marker_cfg = FRAME_MARKER_CFG.copy()
    # frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    # ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    # goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # # Define targets for the arm
    # ee_goal_pose_set_tilted_b = torch.tensor(
    #     [
    #         [0.6, 0.15, 0.3, 0.0, 0.92387953, 0.0, 0.38268343],
    #         [0.6, -0.3, 0.3, 0.0, 0.92387953, 0.0, 0.38268343],
    #         [0.8, 0.0, 0.5, 0.0, 0.92387953, 0.0
    # , 0.38268343],
    #     ],
    #     device=sim.device,
    # )
    # ee_goal_wrench_set_tilted_task = torch.tensor(
    #     [
    #         [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
    #         [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
    #         [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
    #     ],
    #     device=sim.device,
    # )
    # kp_set_task = torch.tensor(
    #     [
    #         [360.0, 360.0, 360.0, 360.0, 360.0, 360.0],
    #         [420.0, 420.0, 420.0, 420.0, 420.0, 420.0],
    #         [320.0, 320.0, 320.0, 320.0, 320.0, 320.0],
    #     ],
    #     device=sim.device,
    # )
    # ee_target_set = torch.cat([ee_goal_pose_set_tilted_b, ee_goal_wrench_set_tilted_task, kp_set_task], dim=-1)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    two_pi = 6.283185307179586
    amp_rad = abs(args_cli.sin_amp_deg) * (3.141592653589793 / 180.0)
    freq_hz = max(0.0, args_cli.sin_freq_hz)

    # Update buffers before first step
    robot.update(dt=sim_dt)
    # 记录初始姿态，未被命令的关节保持此角度
    hold_q_full_init = robot.data.joint_pos[0].clone()
    # 力控模式下可选：清零关节驱动，避免与外部力控叠加
    if args_cli.mode == "torque" and args_cli.zero_drive_in_torque:
        try:
            import torch as _t
            zero = _t.zeros_like(hold_q_full_init)
            if hasattr(robot, 'set_joint_drive_property'):
                robot.set_joint_drive_property(stiffness=zero, damping=zero)
                print("[INFO] torque 模式：已将关节驱动 stiffness/damping 清零")
        except Exception as _e:
            print(f"[WARN] 清零驱动失败：{_e}")

    # Soft limits and centers for selected joints
    soft_limits = robot.data.soft_joint_pos_limits[:, arm_joint_ids_list, :]   # [N, dof_sel, 2]
    joint_centers = torch.mean(soft_limits, dim=-1)                            # [N, dof_sel]

    # get the updated states (initial) — 非OSC模式下可直接从robot.data读取关节状态

    # ROS2 订阅：接收 [[name, angle], ...] 字符串命令
    ros2_enabled = False
    latest_cmd = {"stamp": 0.0, "targets": None}
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        from std_msgs.msg import String as RosString
        from sensor_msgs.msg import JointState as RosJointState
        ros2_enabled = True
        # 简单内嵌 Node（与 Isaac 同线程轮询）
        class _CmdNode(Node):
            def __init__(self):
                super().__init__('joint_cmd_listener')
                self.sub = self.create_subscription(RosString, args_cli.ros2_cmd_topic, self._cb, 10)
                # JointState 发布器
                try:
                    # QoS 配置
                    qos = None
                    if args_cli.ros2_joint_state_qos == "reliable":
                        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
                    elif args_cli.ros2_joint_state_qos == "best_effort":
                        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
                    else:  # sensor
                        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
                    self.pub_js = self.create_publisher(RosJointState, args_cli.ros2_joint_state_topic, qos)
                except Exception:
                    self.pub_js = None
                # 缓存（由仿真循环维护）
                self._js_cache = {
                    "names": [],
                    "q": None,
                    "dq": None,
                    "tau": None,
                }
                self._js_lock = threading.Lock()
                # 定时器：按频率发布（独立于仿真步进）
                self._js_timer = None
                try:
                    if (not args_cli.ros2_joint_state_threaded) and self.pub_js is not None and float(getattr(args_cli, 'ros2_joint_state_rate', 0.0)) > 0.0:
                        period = 1.0 / float(args_cli.ros2_joint_state_rate)
                        period = max(0.001, period)
                        self._js_timer = self.create_timer(period, self._timer_pub)
                except Exception:
                    self._js_timer = None
            def _cb(self, msg: RosString):
                import json
                try:
                    arr = json.loads(msg.data)
                except Exception:
                    # 尝试 eval 安全解析（不推荐），或简单替换单引号
                    try:
                        data = msg.data.replace("'", '"')
                        arr = json.loads(data)
                    except Exception:
                        self.get_logger().warn('无法解析命令字符串，期望 [[name, angle], ...]')
                        return
                if not isinstance(arr, list):
                    self.get_logger().warn('命令应为列表: [[name, angle], ...]')
                    return
                latest_cmd["stamp"] = float(self.get_clock().now().nanoseconds) * 1e-9
                latest_cmd["targets"] = arr
            def _timer_pub(self):
                # 定时器发布 JointState：读取缓存并发布
                if self.pub_js is None:
                    return
                try:
                    with self._js_lock:
                        names = list(self._js_cache.get("names", []))
                        q = self._js_cache.get("q", None)
                        dq = self._js_cache.get("dq", None)
                        tau = self._js_cache.get("tau", None)
                    if q is None or dq is None:
                        return
                    msg = RosJointState()
                    now = self.get_clock().now()
                    msg.header.stamp = now.to_msg()
                    msg.name = names
                    # 将张量复制到 CPU 列表
                    import torch as _t
                    if isinstance(q, _t.Tensor):
                        q_list = q.detach().cpu().tolist()
                    else:
                        q_list = list(q)
                    if isinstance(dq, _t.Tensor):
                        dq_list = dq.detach().cpu().tolist()
                    else:
                        dq_list = list(dq)
                    if isinstance(tau, _t.Tensor):
                        tau_list = tau.detach().cpu().tolist()
                    elif tau is None:
                        tau_list = []
                    else:
                        tau_list = list(tau)
                    msg.position = q_list
                    msg.velocity = dq_list
                    msg.effort = tau_list
                    self.pub_js.publish(msg)
                except Exception:
                    pass
        if not rclpy.ok():
            rclpy.init(args=None)
        ros_node = _CmdNode()
        # 后台 executor，自旋以驱动 timer 与订阅回调
        _executor = None
        _executor_thread = None
        try:
            _executor = MultiThreadedExecutor(num_threads=2)
            _executor.add_node(ros_node)
            def _spin():
                try:
                    _executor.spin()
                except Exception:
                    pass
            _executor_thread = threading.Thread(target=_spin, daemon=True)
            _executor_thread.start()
            setattr(ros_node, "_has_executor", True)
            # 若启用线程发布，启动线程（即使有 executor，也不使用 timer）
            try:
                if args_cli.ros2_joint_state_threaded and getattr(ros_node, 'pub_js', None) is not None and float(getattr(args_cli, 'ros2_joint_state_rate', 0.0)) > 0.0:
                    period = 1.0 / float(args_cli.ros2_joint_state_rate)
                    period = max(0.001, period)
                    def _thread_pub():
                        import time
                        while True:
                            try:
                                with ros_node._js_lock:
                                    names = list(ros_node._js_cache.get("names", []))
                                    q = ros_node._js_cache.get("q", None)
                                    dq = ros_node._js_cache.get("dq", None)
                                    tau = ros_node._js_cache.get("tau", None)
                                if q is not None and dq is not None:
                                    msg = RosJointState()
                                    now = ros_node.get_clock().now()
                                    msg.header.stamp = now.to_msg()
                                    msg.name = names
                                    import torch as _t
                                    q_list = q.detach().cpu().tolist() if isinstance(q, _t.Tensor) else list(q)
                                    dq_list = dq.detach().cpu().tolist() if isinstance(dq, _t.Tensor) else list(dq)
                                    if isinstance(tau, _t.Tensor):
                                        tau_list = tau.detach().cpu().tolist()
                                    elif tau is None:
                                        tau_list = []
                                    else:
                                        tau_list = list(tau)
                                    msg.position = q_list
                                    msg.velocity = dq_list
                                    msg.effort = tau_list
                                    ros_node.pub_js.publish(msg)
                            except Exception:
                                pass
                            time.sleep(period)
                    _tpub = threading.Thread(target=_thread_pub, daemon=True)
                    _tpub.start()
            except Exception:
                pass
        except Exception:
            _executor = None
            _executor_thread = None
            setattr(ros_node, "_has_executor", False)
            # 若 executor 不可用且 timer 也不可用，启用线程兜底发布（与 timer 同周期）
            try:
                if getattr(ros_node, 'pub_js', None) is not None and float(getattr(args_cli, 'ros2_joint_state_rate', 0.0)) > 0.0:
                    period = 1.0 / float(args_cli.ros2_joint_state_rate)
                    period = max(0.001, period)
                    def _thread_pub():
                        import time
                        while True:
                            try:
                                with ros_node._js_lock:
                                    names = list(ros_node._js_cache.get("names", []))
                                    q = ros_node._js_cache.get("q", None)
                                    dq = ros_node._js_cache.get("dq", None)
                                    tau = ros_node._js_cache.get("tau", None)
                                if q is not None and dq is not None:
                                    msg = RosJointState()
                                    now = ros_node.get_clock().now()
                                    msg.header.stamp = now.to_msg()
                                    msg.name = names
                                    import torch as _t
                                    q_list = q.detach().cpu().tolist() if isinstance(q, _t.Tensor) else list(q)
                                    dq_list = dq.detach().cpu().tolist() if isinstance(dq, _t.Tensor) else list(dq)
                                    if isinstance(tau, _t.Tensor):
                                        tau_list = tau.detach().cpu().tolist()
                                    elif tau is None:
                                        tau_list = []
                                    else:
                                        tau_list = list(tau)
                                    msg.position = q_list
                                    msg.velocity = dq_list
                                    msg.effort = tau_list
                                    ros_node.pub_js.publish(msg)
                            except Exception:
                                pass
                            time.sleep(period)
                    _t = threading.Thread(target=_thread_pub, daemon=True)
                    _t.start()
            except Exception:
                pass
    except Exception as _e:
        print(f"[warn] ROS2 订阅初始化失败：{_e}")
        ros2_enabled = False

    # 在每个关节的软限位 [lower, upper] 内做小幅振荡：target = lower + A * 0.5*(1+sin(…))
    lower = soft_limits[..., 0]
    upper = soft_limits[..., 1]
    t = 0.0
    print(f"[info] 正弦抖动: 振幅={args_cli.sin_amp_deg:.1f} deg (~{amp_rad:.3f} rad), 频率={freq_hz:.2f} Hz")
    # JointState 发布频率与缓存
    js_rate_hz = float(getattr(args_cli, "ros2_joint_state_rate", 0.0) or 0.0)
    js_period = 1.0 / js_rate_hz if ros2_enabled and js_rate_hz > 0.0 else None
    last_js_pub_time = -1e9
    last_tau_full = hold_q_full_init.clone() * 0.0  # 最近一次力矩（默认零）

    # Simulation loop
    while simulation_app.is_running():
        # ROS2 spin（非阻塞）
        try:
            if ros2_enabled and not getattr(ros_node, "_has_executor", False):
                import rclpy
                rclpy.spin_once(ros_node, timeout_sec=0.0)
        except Exception:
            pass

        # 更新状态
        # 非OSC模式：直接读取所选关节的当前位姿与速度
        joint_pos = robot.data.joint_pos[:, arm_joint_ids_list]
        joint_vel = robot.data.joint_vel[:, arm_joint_ids_list]

        # 若接收到 ROS2 命令，则构建全关节位置目标：命令关节=指定角度，其余=初始角度保持
        if latest_cmd["targets"] is not None:
            try:
                # 基于初始角度作为未命令关节的保持参考
                hold_q_full = hold_q_full_init.clone()
                name_to_idx = {n: i for i, n in enumerate(robot.joint_names)}
                deg2rad = 3.141592653589793 / 180.0
                for item in latest_cmd["targets"]:
                    if not isinstance(item, (list, tuple)) or len(item) != 2:
                        continue
                    name, val = item[0], item[1]
                    if name not in name_to_idx:
                        continue
                    try:
                        v = float(val)
                        if args_cli.ros2_cmd_unit == "deg":
                            v = v * deg2rad
                        hold_q_full[name_to_idx[name]] = v
                    except Exception:
                        continue
                # 夹紧并写入
                lower_full = robot.data.soft_joint_pos_limits[0, :, 0]
                upper_full = robot.data.soft_joint_pos_limits[0, :, 1]
                import torch as _t
                cmd_targets_clamped = _t.clamp(hold_q_full, min=lower_full, max=upper_full)
                cmd_targets_all = cmd_targets_clamped.unsqueeze(0).expand(scene.num_envs, -1).contiguous()
                robot.set_joint_position_target(cmd_targets_all)
                robot.write_data_to_sim()
                # 短路后续速度控制，直接推进一步
                sim.step(render=True)
                robot.update(sim_dt)
                scene.update(sim_dt)
                # 若启用 ROS2 定时器发布：更新缓存；否则按步发布一次
                try:
                    # 更新缓存供 timer 使用
                    if ros2_enabled and hasattr(ros_node, '_js_lock'):
                        with ros_node._js_lock:
                            ros_node._js_cache["names"] = js_names
                            ros_node._js_cache["q"] = robot.data.joint_pos[0][js_idx_list].clone()
                            ros_node._js_cache["dq"] = robot.data.joint_vel[0][js_idx_list].clone()
                            ros_node._js_cache["tau"] = (last_tau_full[js_idx_list].clone() if isinstance(last_tau_full, torch.Tensor) else last_tau_full)
                    # 若没有 timer，则按步发布（限频）
                    if (
                        ros2_enabled
                        and js_period is not None
                        and getattr(ros_node, 'pub_js', None) is not None
                        and getattr(ros_node, '_js_timer', None) is None
                        and not args_cli.ros2_joint_state_threaded
                    ):
                        now = ros_node.get_clock().now()
                        now_s = float(now.nanoseconds) * 1e-9
                        if now_s - last_js_pub_time >= js_period:
                            msg = RosJointState()
                            msg.header.stamp = now.to_msg()
                            msg.name = js_names
                            q_full = robot.data.joint_pos[0][js_idx_list].detach().cpu().tolist()
                            dq_full = robot.data.joint_vel[0][js_idx_list].detach().cpu().tolist()
                            msg.position = q_full
                            msg.velocity = dq_full
                            if isinstance(last_tau_full, torch.Tensor):
                                msg.effort = last_tau_full[js_idx_list].detach().cpu().tolist()
                            else:
                                msg.effort = []
                            ros_node.pub_js.publish(msg)
                            last_js_pub_time = now_s
                except Exception:
                    pass
                t += float(sim_dt)
                continue
            except Exception:
                pass

        # 若未收到 ROS2 命令，依据模式执行控制
        if args_cli.mode == "osc":
            # 如果你启用了上面的OSC实例化代码，这里可以改为使用OSC力矩控制。
            # 由于本脚本中OSC默认被注释掉，'osc'模式将不执行任何力矩指令。
            pass
        elif args_cli.mode == "torque":
            # 力矩 PD：目标为初始角（未命令关节保持初始角度）
            try:
                q = robot.data.joint_pos[0]
                dq = robot.data.joint_vel[0]
                q_des = hold_q_full_init.clone()
                # 限位夹紧
                lower_full = robot.data.soft_joint_pos_limits[0, :, 0]
                upper_full = robot.data.soft_joint_pos_limits[0, :, 1]
                q_des = torch.clamp(q_des, min=lower_full, max=upper_full)
                kp = float(args_cli.torque_kp)
                kd = float(args_cli.torque_kd)
                tau = kp * (q_des - q) - kd * dq
                limit = float(args_cli.effort_limit)
                tau = torch.clamp(tau, -limit, limit)
                # 扩展到所有环境并下发
                tau_all = tau.unsqueeze(0).expand(scene.num_envs, -1).contiguous()
                robot.set_joint_effort_target(tau_all)
                robot.write_data_to_sim()
                # 记录最近一次力矩
                last_tau_full = tau
            except Exception:
                pass
        else:
            # 速度PD模式：固定外展关节为0，其余屈伸关节目标为joint_target
            # 为避免无谓运动，这里在 vel 模式下当 amp=0 时不再移动（保持当前）
            if freq_hz > 0.0 and amp_rad > 0.0:
                phase = 0.5 * (1.0 + torch.sin(torch.tensor(two_pi * freq_hz * t, device=sim.device)))
                joint_target = lower.clone()
                if flex_idxs:
                    dof_range = (upper - lower)
                    amp_tensor = torch.clamp_min(torch.minimum(0.25 * dof_range, torch.tensor(amp_rad, device=sim.device)), 0.0)
                    joint_target[:, flex_idxs] = lower[:, flex_idxs] + amp_tensor[:, flex_idxs] * phase
                joint_target = torch.clamp(joint_target, min=lower, max=upper)
                pos_target_sel = torch.zeros_like(joint_target)
                if flex_idxs:
                    pos_target_sel[:, flex_idxs] = joint_target[:, flex_idxs]
                pos_target_sel = torch.clamp(pos_target_sel, min=lower, max=upper)
                kp_v = 8.0
                kd_v = 0.2
                vel_cmd_sel = kp_v * (pos_target_sel - joint_pos) - kd_v * joint_vel
                vel_cmd_full = torch.zeros(scene.num_envs, robot.num_joints, device=sim.device)
                vel_cmd_full[:, arm_joint_ids_list] = vel_cmd_sel
                robot.set_joint_velocity_target(vel_cmd_full)
                robot.write_data_to_sim()
            # 非力控时 effort 记为0
            try:
                last_tau_full = last_tau_full * 0.0
            except Exception:
                pass
        # 在步进前更新 JointState 缓存，供 500Hz 定时器读取
        try:
            if ros2_enabled and hasattr(ros_node, '_js_lock'):
                with ros_node._js_lock:
                    ros_node._js_cache["names"] = js_names
                    ros_node._js_cache["q"] = robot.data.joint_pos[0][js_idx_list].clone()
                    ros_node._js_cache["dq"] = robot.data.joint_vel[0][js_idx_list].clone()
                    ros_node._js_cache["tau"] = (last_tau_full[js_idx_list].clone() if isinstance(last_tau_full, torch.Tensor) else last_tau_full)
        except Exception:
            pass
        # step
        sim.step(render=True)
        robot.update(sim_dt)
        scene.update(sim_dt)
        # 若没有 timer，则按步发布（限频）
        try:
            if (
                ros2_enabled
                and js_period is not None
                and getattr(ros_node, 'pub_js', None) is not None
                and getattr(ros_node, '_js_timer', None) is None
                and not args_cli.ros2_joint_state_threaded
            ):
                now = ros_node.get_clock().now()
                now_s = float(now.nanoseconds) * 1e-9
                if now_s - last_js_pub_time >= js_period:
                    msg = RosJointState()
                    msg.header.stamp = now.to_msg()
                    msg.name = js_names
                    q_full = robot.data.joint_pos[0][js_idx_list].detach().cpu().tolist()
                    dq_full = robot.data.joint_vel[0][js_idx_list].detach().cpu().tolist()
                    msg.position = q_full
                    msg.velocity = dq_full
                    if isinstance(last_tau_full, torch.Tensor):
                        msg.effort = last_tau_full[js_idx_list].detach().cpu().tolist()
                    else:
                        msg.effort = []
                    ros_node.pub_js.publish(msg)
                    last_js_pub_time = now_s
        except Exception:
            pass
        t += float(sim_dt)
        # reset every 500 steps
        # if count % 500 == 0:
        #     # reset joint state to default
        #     default_joint_pos = robot.data.default_joint_pos.clone()
        #     default_joint_vel = robot.data.default_joint_vel.clone()
        #     robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
        #     robot.set_joint_effort_target(zero_joint_efforts)  # Set zero torques in the initial step
        #     robot.write_data_to_sim()
        #     robot.reset()
        #     # reset contact sensor
        #     contact_forces.reset()
        #     # reset target pose
        #     robot.update(sim_dt)
        #     _, _, _, ee_pose_b, _, _, _, _, _, _ = update_states(
        #         sim, scene, robot, ee_frame_idx, arm_joint_ids, contact_forces
        #     )  # at reset, the jacobians are not updated to the latest state
        #     command, ee_target_pose_b, ee_target_pose_w, current_goal_idx = update_target(
        #         sim, scene, osc, root_pose_w, ee_target_set, current_goal_idx
        #     )
        #     # set the osc command
        #     osc.reset()
        #     command, task_frame_pose_b = convert_to_task_frame(osc, command=command, ee_target_pose_b=ee_target_pose_b)
        #     osc.set_command(command=command, current_ee_pose_b=ee_pose_b, current_task_frame_pose_b=task_frame_pose_b)
        # else:
        #     # get the updated states
        #     (
        #         jacobian_b,
        #         mass_matrix,
        #         gravity,
        #         ee_pose_b,
        #         ee_vel_b,
        #         root_pose_w,
        #         ee_pose_w,
        #         ee_force_b,
        #         joint_pos,
        #         joint_vel,
        #     ) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids, contact_forces)
        #     # compute the joint commands
        #     joint_efforts = osc.compute(
        #         jacobian_b=jacobian_b,
        #         current_ee_pose_b=ee_pose_b,
        #         current_ee_vel_b=ee_vel_b,
        #         current_ee_force_b=ee_force_b,
        #         mass_matrix=mass_matrix,
        #         gravity=gravity,
        #         current_joint_pos=joint_pos,
        #         current_joint_vel=joint_vel,
        #         nullspace_joint_pos_target=joint_centers,
        #     )
        #     # apply actions
        #     robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
        #     robot.write_data_to_sim()

        # # update marker positions
        # ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        # goal_marker.visualize(ee_target_pose_w[:, 0:3], ee_target_pose_w[:, 3:7])

        # # perform step
        # sim.step(render=True)
        # # update robot buffers
        # robot.update(sim_dt)
        # # update buffers
        # scene.update(sim_dt)
        # # update sim-time
        # count += 1


# Update robot states
def update_states(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    ee_frame_idx: int,
    arm_joint_ids: list[int],
    contact_forces,
):
    """Update the robot states.

    Args:
        sim: (SimulationContext) Simulation context.
        scene: (InteractiveScene) Interactive scene.
        robot: (Articulation) Robot articulation.
        ee_frame_idx: (int) End-effector frame index.
        arm_joint_ids: (list[int]) Arm joint indices.
        contact_forces: (ContactSensor) Contact sensor.

    Returns:
        jacobian_b (torch.tensor): Jacobian in the body frame.
        mass_matrix (torch.tensor): Mass matrix.
        gravity (torch.tensor): Gravity vector.
        ee_pose_b (torch.tensor): End-effector pose in the body frame.
        ee_vel_b (torch.tensor): End-effector velocity in the body frame.
        root_pose_w (torch.tensor): Root pose in the world frame.
        ee_pose_w (torch.tensor): End-effector pose in the world frame.
        ee_force_b (torch.tensor): End-effector force in the body frame.
        joint_pos (torch.tensor): The joint positions.
        joint_vel (torch.tensor): The joint velocities.

    Raises:
        ValueError: Undefined target_type.
    """
    # obtain dynamics related quantities from simulation
    ee_jacobi_idx = ee_frame_idx - 1
    jacobian_w = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]
    gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]
    # Convert the Jacobian from world to root frame
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    # Compute current pose of the end-effector
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_w = robot.data.body_pos_w[:, ee_frame_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_frame_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    root_pose_w = torch.cat([root_pos_w, root_quat_w], dim=-1)
    ee_pose_w = torch.cat([ee_pos_w, ee_quat_w], dim=-1)
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    # Compute the current velocity of the end-effector
    ee_vel_w = robot.data.body_vel_w[:, ee_frame_idx, :]  # Extract end-effector velocity in the world frame
    root_vel_w = robot.data.root_vel_w  # Extract root velocity in the world frame
    relative_vel_w = ee_vel_w - root_vel_w  # Compute the relative velocity in the world frame
    ee_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])  # From world to root frame
    ee_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    # Calculate the contact force
    ee_force_w = torch.zeros(scene.num_envs, 3, device=sim.device)
    sim_dt = sim.get_physics_dt()
    contact_forces.update(sim_dt)  # update contact sensor
    # Calculate the contact force by averaging over last four time steps (i.e., to smoothen) and
    # taking the max of three surfaces as only one should be the contact of interest
    ee_force_w, _ = torch.max(torch.mean(contact_forces.data.net_forces_w_history, dim=1), dim=1)

    # This is a simplification, only for the sake of testing.
    ee_force_b = ee_force_w

    # Get joint positions and velocities
    joint_pos = robot.data.joint_pos[:, arm_joint_ids]
    joint_vel = robot.data.joint_vel[:, arm_joint_ids]

    return (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        ee_force_b,
        joint_pos,
        joint_vel,
    )


# Update the target commands
# def update_target(
#     sim: sim_utils.SimulationContext,
#     scene: InteractiveScene,
#     osc: OperationalSpaceController,
#     root_pose_w: torch.tensor,
#     ee_target_set: torch.tensor,
#     current_goal_idx: int,
# ):
#     """Update the targets for the operational space controller.

#     Args:
#         sim: (SimulationContext) Simulation context.
#         scene: (InteractiveScene) Interactive scene.
#         osc: (OperationalSpaceController) Operational space controller.
#         root_pose_w: (torch.tensor) Root pose in the world frame.
#         ee_target_set: (torch.tensor) End-effector target set.
#         current_goal_idx: (int) Current goal index.

#     Returns:
#         command (torch.tensor): Updated target command.
#         ee_target_pose_b (torch.tensor): Updated target pose in the body frame.
#         ee_target_pose_w (torch.tensor): Updated target pose in the world frame.
#         next_goal_idx (int): Next goal index.

#     Raises:
#         ValueError: Undefined target_type.
#     """

#     # update the ee desired command
#     command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
#     command[:] = ee_target_set[current_goal_idx]

#     # update the ee desired pose
#     ee_target_pose_b = torch.zeros(scene.num_envs, 11, device=sim.device)
#     for target_type in osc.cfg.target_types:
#         if target_type == "pose_abs":
#             ee_target_pose_b[:] = command[:, :7]
#         elif target_type == "wrench_abs":
#             pass  # ee_target_pose_b could stay at the root frame for force control, what matters is ee_target_b
#         else:
#             raise ValueError("Undefined target_type within update_target().")

#     # update the target desired pose in world frame (for marker)
#     ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
#         root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
#     )
#     ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)

#     next_goal_idx = (current_goal_idx + 1) % len(ee_target_set)

#     return command, ee_target_pose_b, ee_target_pose_w, next_goal_idx


# # Convert the target commands to the task frame
# def convert_to_task_frame(osc: OperationalSpaceController, command: torch.tensor, ee_target_pose_b: torch.tensor):
#     """Converts the target commands to the task frame.

#     Args:
#         osc: OperationalSpaceController object.
#         command: Command to be converted.
#         ee_target_pose_b: Target pose in the body frame.

#     Returns:
#         command (torch.tensor): Target command in the task frame.
#         task_frame_pose_b (torch.tensor): Target pose in the task frame.

#     Raises:
#         ValueError: Undefined target_type.
#     """
#     command = command.clone()
#     task_frame_pose_b = ee_target_pose_b.clone()

#     cmd_idx = 0
#     for target_type in osc.cfg.target_types:
#         if target_type == "pose_abs":
#             command[:, :3], command[:, 3:7] = subtract_frame_transforms(
#                 task_frame_pose_b[:, :3], task_frame_pose_b[:, 3:], command[:, :3], command[:, 3:7]
#             )
#             cmd_idx += 7
#         elif target_type == "wrench_abs":
#             # These are already defined in target frame for ee_goal_wrench_set_tilted_task (since it is
#             # easier), so not transforming
#             cmd_idx += 6
#         else:
#             raise ValueError("Undefined target_type within _convert_to_task_frame().")

#     return command, task_frame_pose_b


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()