#!/usr/bin/env python3
"""使用 IsaacLab 的 sim_utils/prim_utils，在地面上于原点与 (10,10) 生成两个蓝色立方体。"""

import argparse

from isaaclab.app import AppLauncher

# 解析并启动 App（与仓库其他脚本保持一致）
parser = argparse.ArgumentParser(description="Spawn two blue cubes at origin and (10,10)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext


def design_scene():
	"""创建地面、灯光与两个蓝色立方体。"""
	# 地面
	cfg_ground = sim_utils.GroundPlaneCfg()
	cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
	# 灯光
	cfg_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
	cfg_light.func("/World/Light", cfg_light)

	# 父级 Xform
	prim_utils.create_prim("/World/Objects", "Xform")

	# 立方体配置（刚体可省略，这里只设置可视材质为蓝色）
	cube_cfg = sim_utils.CuboidCfg(
		size=(1.0, 1.0, 1.0),
		visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
	)
	# 原点：z=0.5 让立方体底部落在地面上
	cube_cfg.func("/World/Objects/Cube_Origin", cube_cfg, translation=(0.0, 0.0, 0.5))
	# (10,10)：同理放置
	cube_cfg.func("/World/Objects/Cube_10_10", cube_cfg, translation=(10.0, 10.0, 0.5))
    # # 你的模型 USD 路径（本地或 Nucleus）
    # # 例1：本地文件
    # usd_path = "/home/yourname/models/my_robot.usd"
    # # 例2：Nucleus 服务器（如果已连接）
    # # usd_path = "omniverse://localhost/Nucleus/Assets/MyRobot/my_robot.usd"

    # # 如果只做静态视觉，可直接加载；若需要碰撞/刚体，请添加 rigid_props/collision_props
    # model_cfg = sim_utils.UsdFileCfg(
    #     usd_path=usd_path,
    #     scale=(1.0, 1.0, 1.0),          # 如模型单位是 cm 而你想用 m，可改为 (0.01, 0.01, 0.01)
    #     make_instanceable=False,       # 多实例大量复制时可设 True
    #     # 可选：绑定一个统一的预览材质（若原模型没有材质或想覆盖）
    #     # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.8)),
    #     # 可选：添加刚体性质和碰撞
    #     # rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #     # collision_props=sim_utils.CollisionPropertiesCfg(),
    # )
    # # 放置到世界坐标 (x,y,z) 与姿态 orientation=(qx, qy, qz, qw)
    # model_cfg.func("/World/MyModel", model_cfg, translation=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0))


def main():
	# 初始化仿真上下文
	sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
	sim = SimulationContext(sim_cfg)
	# 相机视角
	sim.set_camera_view(eye=[12.0, 12.0, 6.0], target=[5.0, 5.0, 0.5])
	# 场景构建
	design_scene()
	# 进入仿真
	sim.reset()
	print("[INFO]: Setup complete... Blue cubes spawned at (0,0) and (10,10).")
	while simulation_app.is_running():
		sim.step()


if __name__ == "__main__":
	main()
	simulation_app.close()
