import argparse
from isaaclab.app import AppLauncher
from isaaclab.sim import simulationCfg,simulationContext
parser = argparse.ArgumentParser(description="Tutorial: Createing an enmpty stage.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
def main():
    sim_cfg = simulationCfg(dt=0.01)
    sim = simulationContext(sim_cfg)
    sim.set_camera_view(position=(2.5, 2.5, 2.5), target=(0, 0, 0))
    sim.reset()
    print("Simulation started")
    while simulation_app.is_running():
        sim.step()
if __name__ == "__main__":
    main()
    simulation_app.close()