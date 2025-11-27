import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os

# Resolve assets path relative to this module. Use absolute paths so the runtime can open USD files reliably.
module_dir = os.path.dirname(__file__)
DEFAULT_ASSETS_DIR = os.path.abspath(os.path.join(module_dir, "..", "assets"))

# Candidate USD locations (module-relative first, then known alternate locations in the workspace).
HI_USD_CANDIDATES = [
    os.path.join(DEFAULT_ASSETS_DIR, "usd", "hi", "hi", "hi.usd"),
    os.path.abspath(os.path.join(module_dir, "..", "..", "source", "sim_1", "assets", "usd", "hi", "hi", "hi.usd")),
    os.path.abspath(os.path.join(module_dir, "..", "..", "sim_1", "assets", "usd", "hi", "hi", "hi.usd")),
]

# Pick the first existing candidate, otherwise keep the first candidate (so the error message shows expected path).
selected_usd = None
for p in HI_USD_CANDIDATES:
    if os.path.exists(p):
        selected_usd = p
        break
if selected_usd is None:
    # fallback to default (non-existing) so the spawn code will raise the same FileNotFoundError but with a clear path
    selected_usd = HI_USD_CANDIDATES[0]

ISAAC_ASSET_DIR = os.path.dirname(selected_usd)
# ISAAC_ASSET_DIR = os.path.join(os.path.dirname(__file__), "../assets")
HI_CFG = ArticulationCfg(
    prim_path="/World/envs/env_0/Robot",  # 实例化到 /World 下，物理属性和 Weld 约束可生效
    spawn=sim_utils.UsdFileCfg(
        usd_path=selected_usd,  # 指向新的 USD 文件，根 prim 为 Robot
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
            ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.60),#0.46
        joint_pos={
            "waist_joint": 0.0,
            "l_hip_pitch_joint": 0.0,
            "r_hip_pitch_joint": 0.0,
            "l_hip_roll_joint": 0.0,
            "r_hip_roll_joint": 0.0,
            "l_thigh_joint": 0.0,
            "r_thigh_joint": 0.0,
            "l_calf_joint": 0.0,
            "r_calf_joint": 0.0,
            "l_ankle_pitch_joint": 0.0,
            "r_ankle_pitch_joint": 0.0,
            "l_ankle_roll_joint": 0.0,
            "r_ankle_roll_joint": 0.0,
            # 上肢关节
            "l_shoulder_pitch_joint": 0.0,
            "r_shoulder_pitch_joint": 0.0,
            "l_shoulder_roll_joint": 0.0,
            "r_shoulder_roll_joint": 0.0,
            "l_upper_arm_joint": 0.0,
            "r_upper_arm_joint": 0.0,
            "l_elbow_joint": 0.0,
            "r_elbow_joint": 0.0,
            "l_wrist_joint": 0.0,
            "r_wrist_joint": 0.0,
        },
        joint_vel={
            "waist_joint": 0.0,
            "l_hip_pitch_joint": 0.0,
            "r_hip_pitch_joint": 0.0,
            "l_hip_roll_joint": 0.0,
            "r_hip_roll_joint": 0.0,
            "l_thigh_joint": 0.0,
            "r_thigh_joint": 0.0,
            "l_calf_joint": 0.0,
            "r_calf_joint": 0.0,
            "l_ankle_pitch_joint": 0.0,
            "r_ankle_pitch_joint": 0.0,
            "l_ankle_roll_joint": 0.0,
            "r_ankle_roll_joint": 0.0,
            # 上肢关节
            "l_shoulder_pitch_joint": 0.0,
            "r_shoulder_pitch_joint": 0.0,
            "l_shoulder_roll_joint": 0.0,
            "r_shoulder_roll_joint": 0.0,
            "l_upper_arm_joint": 0.0,
            "r_upper_arm_joint": 0.0,
            "l_elbow_joint": 0.0,
            "r_elbow_joint": 0.0,
            "l_wrist_joint": 0.0,
            "r_wrist_joint": 0.0,
        },
    ),
    soft_joint_pos_limit_factor = 0.95,
    actuators={
        "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[
                    "waist_joint",
                    "l_hip_pitch_joint",
                    "r_hip_pitch_joint",
                    "l_hip_roll_joint",
                    "r_hip_roll_joint",
                    "l_thigh_joint",
                    "r_thigh_joint",
                    "l_calf_joint",
                    "r_calf_joint",
                    "l_ankle_pitch_joint",
                    "r_ankle_pitch_joint",
                    "l_ankle_roll_joint",
                    "r_ankle_roll_joint",
                    "l_shoulder_pitch_joint",
                    "r_shoulder_pitch_joint",
                    "l_shoulder_roll_joint",
                    "r_shoulder_roll_joint",
                    "l_upper_arm_joint",
                    "r_upper_arm_joint",
                    "l_elbow_joint",
                    "r_elbow_joint",
                    "l_wrist_joint",
                    "r_wrist_joint",
                ],
                effort_limit_sim={
                    "waist_joint": 100.0,
                    "l_hip_pitch_joint": 100.0,
                    "r_hip_pitch_joint": 100.0,
                    "l_hip_roll_joint": 100.0,
                    "r_hip_roll_joint": 100.0,
                    "l_thigh_joint": 100.0,
                    "r_thigh_joint": 100.0,
                    "l_calf_joint": 100.0,
                    "r_calf_joint": 100.0,
                    "l_ankle_pitch_joint": 20.0,
                    "r_ankle_pitch_joint": 20.0,
                    "l_ankle_roll_joint": 20.0,
                    "r_ankle_roll_joint": 20.0,
                    # 上肢关节
                    "l_shoulder_pitch_joint": 60.0,
                    "r_shoulder_pitch_joint": 60.0,
                    "l_shoulder_roll_joint": 60.0,
                    "r_shoulder_roll_joint": 60.0,
                    "l_upper_arm_joint": 50.0,
                    "r_upper_arm_joint": 50.0,
                    "l_elbow_joint": 40.0,
                    "r_elbow_joint": 40.0,
                    "l_wrist_joint": 20.0,
                    "r_wrist_joint": 20.0,
                },
                velocity_limit_sim={
                    "waist_joint": 20.0,
                    "l_hip_pitch_joint": 20.0,
                    "r_hip_pitch_joint": 20.0,
                    "l_hip_roll_joint": 20.0,
                    "r_hip_roll_joint": 20.0,
                    "l_thigh_joint": 20.0,
                    "r_thigh_joint": 20.0,
                    "l_calf_joint": 20.0,
                    "r_calf_joint": 20.0,
                    "l_ankle_pitch_joint": 20.0,
                    "r_ankle_pitch_joint": 20.0,
                    "l_ankle_roll_joint": 20.0,
                    "r_ankle_roll_joint": 20.0,
                    # 上肢关节
                    "l_shoulder_pitch_joint": 20.0,
                    "r_shoulder_pitch_joint": 20.0,
                    "l_shoulder_roll_joint": 20.0,
                    "r_shoulder_roll_joint": 20.0,
                    "l_upper_arm_joint": 20.0,
                    "r_upper_arm_joint": 20.0,
                    "l_elbow_joint": 20.0,
                    "r_elbow_joint": 20.0,
                    "l_wrist_joint": 20.0,
                    "r_wrist_joint": 20.0,
                },
                stiffness={
                    "waist_joint": 80.0,
                    "l_hip_pitch_joint": 80.0,
                    "r_hip_pitch_joint": 80.0,
                    "l_hip_roll_joint": 80.0,
                    "r_hip_roll_joint": 80.0,
                    "l_thigh_joint": 80.0,
                    "r_thigh_joint": 80.0,
                    "l_calf_joint": 80.0,
                    "r_calf_joint": 80.0,
                    "l_ankle_pitch_joint": 80.0,
                    "r_ankle_pitch_joint": 80.0,
                    "l_ankle_roll_joint": 80.0,
                    "r_ankle_roll_joint": 80.0,
                    # 上肢关节
                    "l_shoulder_pitch_joint": 80.0,
                    "r_shoulder_pitch_joint": 80.0,
                    "l_shoulder_roll_joint": 80.0,
                    "r_shoulder_roll_joint": 80.0,
                    "l_upper_arm_joint": 80.0,
                    "r_upper_arm_joint": 80.0,
                    "l_elbow_joint": 80.0,
                    "r_elbow_joint": 80.0,
                    "l_wrist_joint": 80.0,
                    "r_wrist_joint": 80.0,
                },
                damping={
                    "waist_joint": 1.0,
                    "l_hip_pitch_joint": 1.0,
                    "r_hip_pitch_joint": 1.0,
                    "l_hip_roll_joint": 1.0,
                    "r_hip_roll_joint": 1.0,
                    "l_thigh_joint": 1.0,
                    "r_thigh_joint": 1.0,
                    "l_calf_joint": 1.0,
                    "r_calf_joint": 1.0,
                    "l_ankle_pitch_joint": 1.0,
                    "r_ankle_pitch_joint": 1.0,
                    "l_ankle_roll_joint": 1.0,
                    "r_ankle_roll_joint": 1.0,
                    # 上肢关节
                    "l_shoulder_pitch_joint": 1.0,
                    "r_shoulder_pitch_joint": 1.0,
                    "l_shoulder_roll_joint": 1.0,
                    "r_shoulder_roll_joint": 1.0,
                    "l_upper_arm_joint": 1.0,
                    "r_upper_arm_joint": 1.0,
                    "l_elbow_joint": 1.0,
                    "r_elbow_joint": 1.0,
                    "l_wrist_joint": 1.0,
                    "r_wrist_joint": 1.0,
                },
        ),
    },
)
