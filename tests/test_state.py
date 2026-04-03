"""
Live state reader for AeroGrasp.

Spins up HoverAviary with GUI, applies hover RPM to all rotors,
and prints the real drone state every N steps.

Run with:
    python tests/test_state.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "environment", "gym-pybullet-drones"))

from src.state import get_drone_state
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync

PRINT_EVERY = 10   # print state every N steps
TOTAL_STEPS = 10  # how long to run

env = HoverAviary(gui=True, obs=ObservationType.KIN, act=ActionType.RPM)
env.reset(seed=0)

# action = 0 means fly at exactly HOVER_RPM (formula: HOVER_RPM * (1 + 0.05 * action))
action = np.zeros((1, 4))

print(f"\nHover RPM: {env.HOVER_RPM:.1f}")
print(f"Running {TOTAL_STEPS} steps, printing every {PRINT_EVERY} steps...\n")
print(f"{'Step':>5}  {'x':>7}  {'y':>7}  {'z':>7}  {'roll':>7}  {'pitch':>7}  {'yaw':>7}  {'speed':>7}  {'r0 RPM':>9}  {'r1 RPM':>9}  {'r2 RPM':>9}  {'r3 RPM':>9}")
print("-" * 115)

start = time.time()

for step in range(TOTAL_STEPS):
    obs, reward, terminated, truncated, info = env.step(action)

    if step % PRINT_EVERY == 0:
        state = get_drone_state(env, drone_index=0)
        print(
            f"{step:>5}  "
            f"{state.x:>+7.3f}  "
            f"{state.y:>+7.3f}  "
            f"{state.z:>+7.3f}  "
            f"{state.roll:>+7.3f}  "
            f"{state.pitch:>+7.3f}  "
            f"{state.yaw:>+7.3f}  "
            f"{state.speed:>7.3f}  "
            f"{state.rotor0_rpm:>9.1f}  "
            f"{state.rotor1_rpm:>9.1f}  "
            f"{state.rotor2_rpm:>9.1f}  "
            f"{state.rotor3_rpm:>9.1f}"
        )

    sync(step, start, env.CTRL_TIMESTEP)

    if terminated or truncated:
        print("\n[INFO] Episode ended, resetting...")
        env.reset(seed=0)

env.close()
print("\nDone.")
