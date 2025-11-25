import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os

# Prefer local assets next to this file. Allow override via FOOTRL_ASSETS_DIR.
_HERE = os.path.dirname(__file__)
_LOCAL_ASSET_DIR = os.path.normpath(os.path.join(_HERE, "../assets"))
_ENV_ASSET_DIR = os.environ.get("FOOTRL_ASSETS_DIR")

def _resolve_usd_path() -> str:
    """Resolve the USD path for G1 with a few robust candidates.

    Candidates (in order):
    1) ${FOOTRL_ASSETS_DIR}/usd/g1/g1_29dof.usd (if env var set)
    2) ../assets/usd/g1/g1_29dof.usd (relative to this file)
    3) ../assets/usd/g1_29dof/g1_29dof.usd (legacy folder name)
    """
    candidates = []
    if _ENV_ASSET_DIR:
        candidates.append(os.path.join(_ENV_ASSET_DIR, "usd", "g1", "g1_29dof.usd"))
    # Local USD candidates (project assets)
    candidates.append(os.path.join(_LOCAL_ASSET_DIR, "usd", "g1", "g1_29dof.usd"))
    # Prefer physics-authored USD first (contains PhysX articulation root), then the base USD
    candidates.append(os.path.join(_LOCAL_ASSET_DIR, "urdf", "g1_description", "g1_29dof", "configuration", "g1_29dof_physics.usd"))
    candidates.append(os.path.join(_LOCAL_ASSET_DIR, "urdf", "g1_description", "g1_29dof", "g1_29dof.usd"))
    # legacy path fallback
    candidates.append(os.path.join(_LOCAL_ASSET_DIR, "usd", "g1_29dof", "g1_29dof.usd"))
    for p in candidates:
        if os.path.isfile(p):
            try:
                # lightweight trace for debugging asset selection
                print(f"[G1_CFG] Using USD asset: {p}")
            except Exception:
                pass
            return p
    # If not found, return the first candidate (will let Isaac report a clear error with path)
    return candidates[0]

# Minimal G1 articulation configuration pointing to the provided USD asset.
# We keep defaults conservative; users can extend actuators/initial joint maps later.
G1_CFG = ArticulationCfg(
    prim_path="/World/envs/env_0/Robot",
    spawn=sim_utils.UsdFileCfg(
    # Use the provided G1 USD
    usd_path=_resolve_usd_path(),
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
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.60),
        # Populate revolute joint names from the G1 URDF. Values default to 0.0.
        joint_pos={
            "left_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.0,
            "left_ankle_pitch_joint": 0.0,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.0,
            "right_ankle_pitch_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
        joint_vel={
            # Default velocities start at 0.0 for all joints
        },
    ),
    soft_joint_pos_limit_factor=0.95,
    # Actuator defaults derived from G1 URDF limits where available.
    # Values chosen conservatively; tune in-sim if needed.
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ],
            effort_limit_sim={
                "left_hip_pitch_joint": 188.0,#88
                "left_hip_roll_joint": 88.0,
                "left_hip_yaw_joint": 88.0,
                "left_knee_joint": 139.0,
                "left_ankle_pitch_joint": 50.0,
                "left_ankle_roll_joint": 50.0,
                "right_hip_pitch_joint": 188.0,
                "right_hip_roll_joint": 88.0,
                "right_hip_yaw_joint": 88.0,
                "right_knee_joint": 139.0,
                "right_ankle_pitch_joint": 50.0,
                "right_ankle_roll_joint": 50.0,
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
                "left_shoulder_pitch_joint": 25.0,
                "left_shoulder_roll_joint": 25.0,
                "left_shoulder_yaw_joint": 25.0,
                "left_elbow_joint": 25.0,
                "left_wrist_roll_joint": 25.0,
                "left_wrist_pitch_joint": 5.0,
                "left_wrist_yaw_joint": 5.0,
                "right_shoulder_pitch_joint": 25.0,
                "right_shoulder_roll_joint": 25.0,
                "right_shoulder_yaw_joint": 25.0,
                "right_elbow_joint": 25.0,
                "right_wrist_roll_joint": 25.0,
                "right_wrist_pitch_joint": 5.0,
                "right_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                # reasonable defaults inspired by HI_CFG
                "left_hip_pitch_joint": 20.0,
                "left_hip_roll_joint": 20.0,
                "left_hip_yaw_joint": 20.0,
                "left_knee_joint": 20.0,
                "left_ankle_pitch_joint": 20.0,
                "left_ankle_roll_joint": 20.0,
                "right_hip_pitch_joint": 20.0,
                "right_hip_roll_joint": 20.0,
                "right_hip_yaw_joint": 20.0,
                "right_knee_joint": 20.0,
                "right_ankle_pitch_joint": 20.0,
                "right_ankle_roll_joint": 20.0,
                "waist_yaw_joint": 20.0,
                "waist_roll_joint": 20.0,
                "waist_pitch_joint": 20.0,
                "left_shoulder_pitch_joint": 20.0,
                "left_shoulder_roll_joint": 20.0,
                "left_shoulder_yaw_joint": 20.0,
                "left_elbow_joint": 20.0,
                "left_wrist_roll_joint": 20.0,
                "left_wrist_pitch_joint": 5.0,
                "left_wrist_yaw_joint": 5.0,
                "right_shoulder_pitch_joint": 20.0,
                "right_shoulder_roll_joint": 20.0,
                "right_shoulder_yaw_joint": 20.0,
                "right_elbow_joint": 20.0,
                "right_wrist_roll_joint": 20.0,
                "right_wrist_pitch_joint": 5.0,
                "right_wrist_yaw_joint": 5.0,
            },
            stiffness={
                "left_hip_pitch_joint": 200.0,
                "left_hip_roll_joint": 150.0,
                "left_hip_yaw_joint": 150.0,
                "left_knee_joint": 200.0,
                "left_ankle_pitch_joint": 20.0,
                "left_ankle_roll_joint": 20.0,
                "right_hip_pitch_joint": 200.0,
                "right_hip_roll_joint": 150.0,
                "right_hip_yaw_joint": 150.0,
                "right_knee_joint": 200.0,
                "right_ankle_pitch_joint": 20.0,
                "right_ankle_roll_joint": 20.0,
                "waist_yaw_joint": 200.0,
                "waist_roll_joint": 200.0,
                "waist_pitch_joint": 200.0,
                "left_shoulder_pitch_joint": 40.0,
                "left_shoulder_roll_joint": 40.0,
                "left_shoulder_yaw_joint": 40.0,
                "left_elbow_joint": 40.0,
                "left_wrist_roll_joint": 40.0,
                "left_wrist_pitch_joint": 40.0,
                "left_wrist_yaw_joint": 40.0,
                "right_shoulder_pitch_joint": 40.0,
                "right_shoulder_roll_joint": 40.0,
                "right_shoulder_yaw_joint": 40.0,
                "right_elbow_joint": 40.0,
                "right_wrist_roll_joint": 40.0,
                "right_wrist_pitch_joint": 40.0,
                "right_wrist_yaw_joint": 40.0,
            },
            damping={
                # Increased damping to reduce oscillations
                "left_hip_pitch_joint": 40, #5.0,
                "left_hip_roll_joint": 5.0,
                "left_hip_yaw_joint": 5.0,
                "left_knee_joint": 15,#5.0,
                "left_ankle_pitch_joint": 2.0,
                "left_ankle_roll_joint": 2.0,
                "right_hip_pitch_joint": 15,#5.0,
                "right_hip_roll_joint": 5.0,
                "right_hip_yaw_joint": 5.0,
                "right_knee_joint": 15,#5.0,
                "right_ankle_pitch_joint": 2.0,
                "right_ankle_roll_joint": 2.0,
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0,
                "left_shoulder_pitch_joint": 10.0,
                "left_shoulder_roll_joint": 10.0,
                "left_shoulder_yaw_joint": 10.0,
                "left_elbow_joint": 10.0,
                "left_wrist_roll_joint": 10.0,
                "left_wrist_pitch_joint": 10.0,
                "left_wrist_yaw_joint": 10.0,
                "right_shoulder_pitch_joint": 10.0,
                "right_shoulder_roll_joint": 10.0,
                "right_shoulder_yaw_joint": 10.0,
                "right_elbow_joint": 10.0,
                "right_wrist_roll_joint": 10.0,
                "right_wrist_pitch_joint": 10.0,
                "right_wrist_yaw_joint": 10.0,
            },
        )
    },
)
