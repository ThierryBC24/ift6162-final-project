import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from mpc_ship_nav.charts.environment import ChartEnvironment
from mpc_ship_nav.sim.engine import SimLog, SimConfig
from mpc_ship_nav.dynamics.vessel import Vessel, VesselState, VesselParams
from tqdm import tqdm
import math

class SimulateTraj:
    def __init__(self, simlog: SimLog, ):
        self.n_candidate = len(simlog.masks_per_snapshot[0])
        self.horizon = len(simlog.trajs_per_snapshot[0][0])
        self.simlog = simlog
        
    def get_traj_per_snapshot(self, time_index: int):
        return self.simlog.trajs_per_snapshot[time_index]
    
    def get_colors_per_snapshot(self, time_index: int):
        mask = self.simlog.masks_per_snapshot[time_index]
        u_argmin = self.simlog.controls_idx[time_index]
        return mask, u_argmin
    
    
        
        
    
    
            
    