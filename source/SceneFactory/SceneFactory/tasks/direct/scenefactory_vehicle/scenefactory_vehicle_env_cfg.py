# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class ScenefactoryVehicleEnvCfg(DirectRLEnvCfg):
    """Minimal direct-RL environment for a custom articulated vehicle.

    Expected vehicle structure:
    - One rigid chassis root
    - Four wheel joints
    - Optional front steering joints
    - Optional passive suspension joints
    """

    # env
    decimation = 4
    episode_length_s = 20.0
    action_space = 3  # [throttle, steering, brake]
    # root pose/vel + wheel states + steer states + goal-relative features + ground friction features
    observation_space = 32
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(device="cuda:1", dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=20.0, replicate_physics=False)

    # vehicle
    vehicle_usd_path = "assets/vehicles/custom_vehicle_no_chassis_col.usd"
    vehicle_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Vehicle",
        spawn=sim_utils.UsdFileCfg(usd_path=vehicle_usd_path),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8)),
        actuators={
            # Drive wheels via torque/effort control.
            "wheel_drive": ImplicitActuatorCfg(
                joint_names_expr=[".*wheel.*"],
                stiffness=0.0,
                damping=4.0,
                effort_limit=4000.0,
                velocity_limit=300.0,
            ),
            # Keep suspension passive but damped.
            "suspension": ImplicitActuatorCfg(
                joint_names_expr=[".*suspension.*"],
                stiffness=0.0,
                damping=150.0,
                effort_limit=1000.0,
                velocity_limit=20.0,
            ),
            # Optional steering joints (front knuckles).
            "steering": ImplicitActuatorCfg(
                joint_names_expr=[".*steer.*"],
                stiffness=1200.0,
                damping=120.0,
                effort_limit=20000.0,
                velocity_limit=50.0,
            ),
        },
    )
    # Wheel joint names/expressions in your USD.
    left_wheel_dof_names = [".*front_left_wheel.*", ".*rear_left_wheel.*"]
    right_wheel_dof_names = [".*front_right_wheel.*", ".*rear_right_wheel.*"]
    # Optional steering joints. If empty or unmatched, env uses differential wheel torque steering.
    steer_dof_names = [".*front_left_steer.*", ".*front_right_steer.*"]

    # control mapping from action [-1, 1]
    throttle_action_scale = 120.0  # wheel effort command
    steering_action_scale = 30.5  # radians for steer joints (if present)
    brake_action_scale = 240.0  # opposing wheel effort command
    differential_steer_scale = 0.6  # left/right effort split when steer joints are absent
    # steering control mode
    steering_use_effort_control = True
    steering_kp = 4000.0
    steering_kd = 200.0
    steering_effort_limit = 20000.0

    # start-goal navigation task
    start_random_x_range = (-20.0, 20.0)
    start_random_y_range = (-20.0, 20.0)
    goal_random_x_range = (-20.0, 20.0)
    goal_random_y_range = (-20.0, 20.0)
    min_start_goal_distance = 3.0
    min_goal_forward_displacement = 0.1
    goal_border_margin = 2.0
    start_corner_margin = 2.0
    goal_reach_threshold = 0.75
    # reward scales
    rew_progress_scale = 12.0
    rew_speed_to_goal_scale = 1.0
    rew_goal_distance_scale = 0.6
    rew_heading_alignment_scale = 1.5
    rew_goal_reached_bonus = 50.0
    rew_stall_scale = 0.8
    stall_speed_threshold = 0.25
    stall_distance_threshold = 1.5
    rew_yaw_rate_scale = 0.0
    rew_action_scale = 0.002
    rew_steer_rate_scale = 0.2
    # boundary visualization
    draw_env_boundaries = True
    boundary_height = 0.03
    boundary_thickness = 0.10
    boundary_color = (0.15, 0.95, 0.35)
    # per-env ground patches and friction
    use_individual_ground_patches = True
    ground_patch_size = (18.0, 18.0)
    ground_patch_thickness = 0.20
    ground_static_friction_range = (0.3, 1.8)
    ground_dynamic_friction_range = (0.2, 1.5)
    # Keep dynamic friction physically close to static friction when randomizing.
    ground_dynamic_to_static_min_ratio = 0.7
    ground_restitution = 0.0

    # reset/termination
    terminate_on_height = True
    min_base_height = 0.05
    max_base_height = 5.0
    randomize_wheel_pos_range = (-0.1, 0.1)
    randomize_wheel_vel_range = (-0.5, 0.5)
    # debug
    debug_log_base_height = False
    debug_log_interval_steps = 10

    def __post_init__(self):
        super().__post_init__()
        base_obs = 13 + 2 * (len(self.left_wheel_dof_names) + len(self.right_wheel_dof_names)) + len(
            self.steer_dof_names
        )
        # goal observation:
        # world-frame goal rel (2) + goal dir (2) + goal dist (1) +
        # body-frame goal rel (2) + heading error sin/cos (2)
        # + ground friction (static, dynamic)
        self.observation_space = base_obs + 11
        # Keep start and goal samples inside each env cell to avoid overlap with neighbor envs.
        nav_bound = 0.45 * float(self.scene.env_spacing)
        self.start_random_x_range = (
            max(self.start_random_x_range[0], -nav_bound),
            min(self.start_random_x_range[1], nav_bound),
        )
        self.start_random_y_range = (
            max(self.start_random_y_range[0], -nav_bound),
            min(self.start_random_y_range[1], nav_bound),
        )
        self.goal_random_x_range = (
            max(self.goal_random_x_range[0], -nav_bound),
            min(self.goal_random_x_range[1], nav_bound),
        )
        self.goal_random_y_range = (
            max(self.goal_random_y_range[0], -nav_bound),
            min(self.goal_random_y_range[1], nav_bound),
        )
        if self.vehicle_usd_path.startswith("<PATH_TO_"):
            self.vehicle_cfg.spawn.usd_path = self.vehicle_usd_path
            return

        configured_path = Path(self.vehicle_usd_path).expanduser()
        if configured_path.is_absolute():
            resolved_path = configured_path
        else:
            repo_root = Path(__file__).resolve().parents[6]
            resolved_path = repo_root / configured_path

        self.vehicle_cfg.spawn.usd_path = str(resolved_path)
