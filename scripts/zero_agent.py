# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--ground_static_friction_min", type=float, default=0.1, help="Min ground static friction.")
parser.add_argument("--ground_static_friction_max", type=float, default=1.8, help="Max ground static friction.")
parser.add_argument("--ground_dynamic_friction_min", type=float, default=0.1, help="Min ground dynamic friction.")
parser.add_argument("--ground_dynamic_friction_max", type=float, default=1.5, help="Max ground dynamic friction.")
parser.add_argument(
    "--fixed_start_position",
    action="store_true",
    default=False,
    help="Use a fixed start position (same local XY for all envs).",
)
parser.add_argument("--start_x", type=float, default=0.0, help="Fixed start x (env-local) when enabled.")
parser.add_argument("--start_y", type=float, default=0.0, help="Fixed start y (env-local) when enabled.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import SceneFactory.tasks  # noqa: F401


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # Zero-agent debug mode: randomize per-env ground friction at spawn.
    if hasattr(env_cfg, "ground_static_friction_range"):
        env_cfg.ground_static_friction_range = (
            args_cli.ground_static_friction_min,
            args_cli.ground_static_friction_max,
        )
    if hasattr(env_cfg, "ground_dynamic_friction_range"):
        env_cfg.ground_dynamic_friction_range = (
            args_cli.ground_dynamic_friction_min,
            args_cli.ground_dynamic_friction_max,
        )
    if args_cli.fixed_start_position:
        if hasattr(env_cfg, "start_random_x_range"):
            env_cfg.start_random_x_range = (args_cli.start_x, args_cli.start_x)
        if hasattr(env_cfg, "start_random_y_range"):
            env_cfg.start_random_y_range = (args_cli.start_y, args_cli.start_y)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    step_count = 0
    debug_print_every = 30
    steer_joint_ids = getattr(env.unwrapped, "_steer_dof_idx", [])
    all_joint_names = getattr(env.unwrapped.vehicle, "joint_names", None)
    if all_joint_names is not None:
        print("[INFO]: Vehicle joint name map:")
        for idx, name in enumerate(all_joint_names):
            print(f"  [{idx}] {name}")
    if steer_joint_ids:
        print(f"[INFO]: Steering joint ids: {steer_joint_ids}")
        if all_joint_names is not None:
            mapped_names = [all_joint_names[i] for i in steer_joint_ids]
            print(f"[INFO]: Steering joint names: {mapped_names}")
    else:
        print("[WARN]: No steering joints found on env.unwrapped.")
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # scripted actions for steering sanity-check:
            # - moderate forward throttle
            # - max-left steering
            # - no braking
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            if actions.shape[-1] >= 1:
                actions[:, 0] = 1.0
            if actions.shape[-1] >= 2:
                actions[:, 1] = -1.0
            if actions.shape[-1] >= 3:
                actions[:, 2] = 0.0
            # apply actions
            env.step(actions)
            if steer_joint_ids and step_count % debug_print_every == 0:
                steer_pos = env.unwrapped.vehicle.data.joint_pos[:, steer_joint_ids]
                steer_cmd = actions[:, 1] if actions.shape[-1] >= 2 else torch.zeros(
                    (actions.shape[0],), device=env.unwrapped.device
                )
                print(
                    "[DEBUG] steer "
                    f"step={step_count} cmd={float(torch.mean(steer_cmd).item()):.3f} "
                    f"pos_mean={float(torch.mean(steer_pos).item()):.3f} "
                    f"pos_min={float(torch.min(steer_pos).item()):.3f} "
                    f"pos_max={float(torch.max(steer_pos).item()):.3f}"
                )
            step_count += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
