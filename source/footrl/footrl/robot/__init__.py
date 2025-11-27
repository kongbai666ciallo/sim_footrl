#from sim_1.assets.usd.high_torque.high_torque import *
from footrl.robot.hi import *
from footrl.robot.g1 import *
import os
import toml

# Conveniences to other module directories via relative paths
ISAAC_ASSET_DIR = os.path.abspath(os.path.dirname(__file__))