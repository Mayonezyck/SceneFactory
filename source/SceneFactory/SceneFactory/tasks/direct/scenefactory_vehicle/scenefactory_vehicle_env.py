# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import sample_uniform

from .scenefactory_vehicle_env_cfg import ScenefactoryVehicleEnvCfg


class ScenefactoryVehicleEnv(DirectRLEnv):
    cfg: ScenefactoryVehicleEnvCfg

    def __init__(self, cfg: ScenefactoryVehicleEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.vehicle_usd_path.startswith("<PATH_TO_"):
            raise ValueError(
                "Set 'vehicle_usd_path' in ScenefactoryVehicleEnvCfg to your articulated vehicle USD before running."
            )
        if not Path(cfg.vehicle_cfg.spawn.usd_path).is_file():
            raise FileNotFoundError(
                f"Vehicle USD not found: '{cfg.vehicle_cfg.spawn.usd_path}'. "
                "Use a repo-relative path like 'assets/Vehicle.usd' or an absolute filesystem path."
            )

        super().__init__(cfg, render_mode, **kwargs)

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._left_wheel_dof_idx = self._find_joint_indices(self.cfg.left_wheel_dof_names, required=True)
        self._right_wheel_dof_idx = self._find_joint_indices(self.cfg.right_wheel_dof_names, required=True)
        self._wheel_dof_idx = sorted(set(self._left_wheel_dof_idx + self._right_wheel_dof_idx))
        self._steer_dof_idx = self._find_joint_indices(self.cfg.steer_dof_names, required=False)
        self._steer_obs_dim = len(self.cfg.steer_dof_names)
        if self._steer_dof_idx:
            print(f"[INFO] Steering joints resolved: {self._steer_dof_idx} (position-target steering enabled)")
        else:
            print("[WARN] No steering joints resolved; using differential wheel torque steering fallback.")
        self._debug_step_counter = 0
        self._start_xy_w = torch.zeros((self.num_envs, 2), device=self.device)
        self._goal_xy_w = torch.zeros((self.num_envs, 2), device=self.device)
        self._prev_goal_dist = torch.zeros((self.num_envs,), device=self.device)
        self._prev_steer_action = torch.zeros((self.num_envs,), device=self.device)
        self._steer_action_delta = torch.zeros((self.num_envs,), device=self.device)

    def _setup_scene(self):
        self.vehicle = Articulation(self.cfg.vehicle_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["vehicle"] = self.vehicle
        if self.cfg.use_individual_ground_patches:
            self._setup_env_ground_patches()
        if self.cfg.draw_env_boundaries:
            self._setup_env_boundaries()
        self._setup_nav_markers()

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)
        # Enforce forward-only throttle.
        self.actions[:, 0] = torch.clamp(self.actions[:, 0], 0.0, 1.0)
        # Brake is engage-only.
        if self.actions.shape[-1] >= 3:
            self.actions[:, 2] = torch.clamp(self.actions[:, 2], 0.0, 1.0)
        steer_now = self.actions[:, 1]
        self._steer_action_delta = torch.abs(steer_now - self._prev_steer_action)
        self._prev_steer_action = steer_now.clone()

    def _apply_action(self) -> None:
        throttle = self.actions[:, 0] * self.cfg.throttle_action_scale
        steer = self.actions[:, 1]
        brake = torch.clamp(self.actions[:, 2], min=0.0) * self.cfg.brake_action_scale
        wheel_vel = self.vehicle.data.joint_vel[:, self._wheel_dof_idx]
        wheel_drive_effort = throttle.unsqueeze(-1).repeat(1, len(self._wheel_dof_idx))
        wheel_brake_effort = brake.unsqueeze(-1).repeat(1, len(self._wheel_dof_idx)) * torch.sign(wheel_vel)
        wheel_effort = wheel_drive_effort - wheel_brake_effort

        if self._steer_dof_idx:
            steer_target = (steer * self.cfg.steering_action_scale).unsqueeze(-1).repeat(1, len(self._steer_dof_idx))
            self.vehicle.set_joint_effort_target(wheel_effort, joint_ids=self._wheel_dof_idx)
            if self.cfg.steering_use_effort_control:
                steer_pos = self.vehicle.data.joint_pos[:, self._steer_dof_idx]
                steer_vel = self.vehicle.data.joint_vel[:, self._steer_dof_idx]
                steer_effort = self.cfg.steering_kp * (steer_target - steer_pos) - self.cfg.steering_kd * steer_vel
                steer_effort = torch.clamp(
                    steer_effort, -self.cfg.steering_effort_limit, self.cfg.steering_effort_limit
                )
                self.vehicle.set_joint_effort_target(steer_effort, joint_ids=self._steer_dof_idx)
            else:
                self.vehicle.set_joint_position_target(steer_target, joint_ids=self._steer_dof_idx)
            return

        # Fallback for vehicles without steering joints: differential wheel effort.
        left_vel = self.vehicle.data.joint_vel[:, self._left_wheel_dof_idx]
        right_vel = self.vehicle.data.joint_vel[:, self._right_wheel_dof_idx]
        left_drive = (throttle * (1.0 - self.cfg.differential_steer_scale * steer)).unsqueeze(-1)
        right_drive = (throttle * (1.0 + self.cfg.differential_steer_scale * steer)).unsqueeze(-1)
        left_effort = left_drive.repeat(1, len(self._left_wheel_dof_idx)) - brake.unsqueeze(-1).repeat(
            1, len(self._left_wheel_dof_idx)
        ) * torch.sign(left_vel)
        right_effort = right_drive.repeat(1, len(self._right_wheel_dof_idx)) - brake.unsqueeze(-1).repeat(
            1, len(self._right_wheel_dof_idx)
        ) * torch.sign(right_vel)
        self.vehicle.set_joint_effort_target(left_effort, joint_ids=self._left_wheel_dof_idx)
        self.vehicle.set_joint_effort_target(right_effort, joint_ids=self._right_wheel_dof_idx)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        root_state = self.vehicle.data.root_state_w
        joint_pos = self.vehicle.data.joint_pos
        joint_vel = self.vehicle.data.joint_vel
        wheel_pos = joint_pos[:, self._wheel_dof_idx]
        wheel_vel = joint_vel[:, self._wheel_dof_idx]
        if self._steer_dof_idx:
            steer_pos = joint_pos[:, self._steer_dof_idx]
        else:
            steer_pos = torch.zeros((self.num_envs, self._steer_obs_dim), dtype=joint_pos.dtype, device=self.device)
        goal_rel = self._goal_xy_w - root_state[:, 0:2]
        goal_dist = torch.linalg.norm(goal_rel, dim=-1, keepdim=True)
        goal_dir = goal_rel / torch.clamp(goal_dist, min=1e-6)
        # Explicit steering cues: goal in body frame + heading error.
        q_w = root_state[:, 3]
        q_x = root_state[:, 4]
        q_y = root_state[:, 5]
        q_z = root_state[:, 6]
        yaw = torch.atan2(2.0 * (q_w * q_z + q_x * q_y), 1.0 - 2.0 * (q_y * q_y + q_z * q_z))
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        goal_rel_body_x = cos_yaw * goal_rel[:, 0] + sin_yaw * goal_rel[:, 1]
        goal_rel_body_y = -sin_yaw * goal_rel[:, 0] + cos_yaw * goal_rel[:, 1]
        goal_rel_body = torch.stack((goal_rel_body_x, goal_rel_body_y), dim=-1)
        heading_err = torch.atan2(goal_rel_body_y, goal_rel_body_x)
        heading_err_sin = torch.sin(heading_err).unsqueeze(-1)
        heading_err_cos = torch.cos(heading_err).unsqueeze(-1)
        if hasattr(self, "_ground_static_friction") and hasattr(self, "_ground_dynamic_friction"):
            ground_static = self._ground_static_friction.unsqueeze(-1)
            ground_dynamic = self._ground_dynamic_friction.unsqueeze(-1)
        else:
            ground_static = torch.ones((self.num_envs, 1), dtype=joint_pos.dtype, device=self.device)
            ground_dynamic = torch.ones((self.num_envs, 1), dtype=joint_pos.dtype, device=self.device)

        obs = torch.cat(
            (
                root_state[:, 0:3],
                root_state[:, 3:7],
                root_state[:, 7:10],
                root_state[:, 10:13],
                wheel_pos,
                wheel_vel,
                steer_pos,
                goal_rel,
                goal_dir,
                goal_dist,
                goal_rel_body,
                heading_err_sin,
                heading_err_cos,
                ground_static,
                ground_dynamic,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        root_state = self.vehicle.data.root_state_w
        pos_xy = self.vehicle.data.root_pos_w[:, 0:2]
        vel_xy = self.vehicle.data.root_lin_vel_w[:, 0:2]
        yaw_rate = torch.abs(self.vehicle.data.root_ang_vel_w[:, 2])
        goal_rel = self._goal_xy_w - pos_xy
        goal_dist = torch.linalg.norm(goal_rel, dim=-1)
        goal_dir = goal_rel / torch.clamp(goal_dist.unsqueeze(-1), min=1e-6)
        speed_to_goal = torch.sum(vel_xy * goal_dir, dim=-1)
        progress = self._prev_goal_dist - goal_dist
        goal_reached = goal_dist < self.cfg.goal_reach_threshold
        q_w = root_state[:, 3]
        q_x = root_state[:, 4]
        q_y = root_state[:, 5]
        q_z = root_state[:, 6]
        yaw = torch.atan2(2.0 * (q_w * q_z + q_x * q_y), 1.0 - 2.0 * (q_y * q_y + q_z * q_z))
        goal_bearing = torch.atan2(goal_rel[:, 1], goal_rel[:, 0])
        heading_error = torch.atan2(torch.sin(goal_bearing - yaw), torch.cos(goal_bearing - yaw))
        heading_alignment = torch.cos(heading_error)
        stalled = (goal_dist > self.cfg.stall_distance_threshold) & (
            torch.abs(speed_to_goal) < self.cfg.stall_speed_threshold
        )
        action_penalty = torch.sum(self.actions**2, dim=-1)
        self._prev_goal_dist = goal_dist
        return (
            self.cfg.rew_progress_scale * progress
            + self.cfg.rew_speed_to_goal_scale * speed_to_goal
            - self.cfg.rew_goal_distance_scale * goal_dist
            + self.cfg.rew_heading_alignment_scale * heading_alignment
            + self.cfg.rew_goal_reached_bonus * goal_reached.float()
            - self.cfg.rew_stall_scale * stalled.float()
            - self.cfg.rew_yaw_rate_scale * yaw_rate
            - self.cfg.rew_action_scale * action_penalty
            - self.cfg.rew_steer_rate_scale * self._steer_action_delta
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        base_height = self.vehicle.data.root_pos_w[:, 2]
        reached_goal = self._prev_goal_dist < self.cfg.goal_reach_threshold
        if self.cfg.debug_log_base_height:
            self._debug_step_counter += 1
            if self._debug_step_counter % max(int(self.cfg.debug_log_interval_steps), 1) == 0:
                h_mean = torch.mean(base_height).item()
                h_min = torch.min(base_height).item()
                h_max = torch.max(base_height).item()
                print(
                    "[DEBUG] base_height "
                    f"step={self._debug_step_counter} mean={h_mean:.4f} min={h_min:.4f} max={h_max:.4f}"
                )
        if self.cfg.terminate_on_height:
            terminated = (base_height < self.cfg.min_base_height) | (base_height > self.cfg.max_base_height) | reached_goal
        else:
            terminated = reached_goal
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.vehicle._ALL_INDICES
        super()._reset_idx(env_ids)

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        default_root_state = self.vehicle.data.default_root_state[env_ids_t].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids_t]
        joint_pos = self.vehicle.data.default_joint_pos[env_ids_t].clone()
        joint_vel = self.vehicle.data.default_joint_vel[env_ids_t].clone()

        start_xy, goal_xy = self._sample_start_goal(env_ids_t)
        self._start_xy_w[env_ids_t] = start_xy
        self._goal_xy_w[env_ids_t] = goal_xy
        goal_vec = goal_xy - start_xy
        default_root_state[:, 0:2] = start_xy
        # Start with a consistent diagonal heading in every env (toward +x,+y).
        yaw = torch.full((len(env_ids_t),), torch.pi / 4.0, device=self.device)
        half_yaw = 0.5 * yaw
        default_root_state[:, 3] = torch.cos(half_yaw)  # qw
        default_root_state[:, 4] = 0.0  # qx
        default_root_state[:, 5] = 0.0  # qy
        default_root_state[:, 6] = torch.sin(half_yaw)  # qz
        self._prev_goal_dist[env_ids_t] = torch.linalg.norm(goal_vec, dim=-1)
        self._prev_steer_action[env_ids_t] = 0.0
        self._steer_action_delta[env_ids_t] = 0.0

        if self._wheel_dof_idx:
            joint_pos[:, self._wheel_dof_idx] += sample_uniform(
                self.cfg.randomize_wheel_pos_range[0],
                self.cfg.randomize_wheel_pos_range[1],
                joint_pos[:, self._wheel_dof_idx].shape,
                joint_pos.device,
            )
            joint_vel[:, self._wheel_dof_idx] += sample_uniform(
                self.cfg.randomize_wheel_vel_range[0],
                self.cfg.randomize_wheel_vel_range[1],
                joint_vel[:, self._wheel_dof_idx].shape,
                joint_vel.device,
            )

        self.vehicle.write_root_pose_to_sim(default_root_state[:, :7], env_ids_t)
        self.vehicle.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids_t)
        self.vehicle.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_t)
        self._reset_nav_markers(env_ids_t)

    def _find_joint_indices(self, patterns: list[str], required: bool) -> list[int]:
        joint_ids: list[int] = []
        for pattern in patterns:
            ids, _ = self.vehicle.find_joints(pattern)
            if ids is None:
                continue
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            joint_ids.extend(ids)
        joint_ids = sorted(set(joint_ids))
        if required and len(joint_ids) == 0:
            raise RuntimeError(
                f"Could not resolve joints for patterns: {patterns}. "
                "Update wheel/steering joint expressions in ScenefactoryVehicleEnvCfg."
            )
        return joint_ids

    def _setup_nav_markers(self) -> None:
        stage = get_current_stage()
        self._start_marker_translate_ops = []
        self._goal_marker_translate_ops = []
        for env_prim_path in self.scene.env_prim_paths:
            start_xf = UsdGeom.Xform.Define(stage, f"{env_prim_path}/start_marker")
            start_geom = UsdGeom.Cube.Define(stage, f"{env_prim_path}/start_marker/geom")
            start_geom.CreateSizeAttr(1.0)
            start_geom.CreateDisplayColorAttr([Gf.Vec3f(0.10, 0.95, 0.25)])
            start_geom_xf = UsdGeom.Xformable(start_geom.GetPrim())
            self._get_or_add_xform_op(start_geom_xf, UsdGeom.XformOp.TypeScale).Set(Gf.Vec3f(0.4, 0.4, 0.05))
            self._start_marker_translate_ops.append(
                self._get_or_add_xform_op(UsdGeom.Xformable(start_xf.GetPrim()), UsdGeom.XformOp.TypeTranslate)
            )

            goal_xf = UsdGeom.Xform.Define(stage, f"{env_prim_path}/goal_marker")
            goal_geom = UsdGeom.Cube.Define(stage, f"{env_prim_path}/goal_marker/geom")
            goal_geom.CreateSizeAttr(1.0)
            goal_geom.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.15, 0.15)])
            goal_geom_xf = UsdGeom.Xformable(goal_geom.GetPrim())
            self._get_or_add_xform_op(goal_geom_xf, UsdGeom.XformOp.TypeScale).Set(Gf.Vec3f(0.45, 0.45, 0.05))
            self._goal_marker_translate_ops.append(
                self._get_or_add_xform_op(UsdGeom.Xformable(goal_xf.GetPrim()), UsdGeom.XformOp.TypeTranslate)
            )

    def _setup_env_ground_patches(self) -> None:
        stage = get_current_stage()
        patch_x, patch_y = self.cfg.ground_patch_size
        patch_z = self.cfg.ground_patch_thickness
        static_f_low, static_f_high = self.cfg.ground_static_friction_range
        dynamic_f_low, dynamic_f_high = self.cfg.ground_dynamic_friction_range
        self._ground_static_friction = torch.ones((self.num_envs,), device=self.device, dtype=torch.float32)
        self._ground_dynamic_friction = torch.ones((self.num_envs,), device=self.device, dtype=torch.float32)

        for env_idx, env_prim_path in enumerate(self.scene.env_prim_paths):
            static_friction = float(torch.empty(1).uniform_(static_f_low, static_f_high).item())
            # Couple dynamic friction to static friction to avoid unrealistic combinations.
            dyn_low = max(dynamic_f_low, static_friction * float(self.cfg.ground_dynamic_to_static_min_ratio))
            dyn_high = min(dynamic_f_high, static_friction)
            if dyn_low > dyn_high:
                dyn_low = dyn_high
            dynamic_friction = float(torch.empty(1).uniform_(dyn_low, dyn_high).item())
            self._ground_static_friction[env_idx] = static_friction
            self._ground_dynamic_friction[env_idx] = dynamic_friction

            # Color map by dynamic friction only: slipperier -> blue, rougher -> darker.
            if abs(dynamic_f_high - dynamic_f_low) < 1e-6:
                roughness = 0.5
            else:
                roughness = (dynamic_friction - dynamic_f_low) / (dynamic_f_high - dynamic_f_low)
                roughness = max(0.0, min(1.0, roughness))
            slippery_blue = (0.16, 0.36, 0.90)
            rough_dark = (0.08, 0.08, 0.08)
            patch_color = Gf.Vec3f(
                float(slippery_blue[0] * (1.0 - roughness) + rough_dark[0] * roughness),
                float(slippery_blue[1] * (1.0 - roughness) + rough_dark[1] * roughness),
                float(slippery_blue[2] * (1.0 - roughness) + rough_dark[2] * roughness),
            )

            # Each env gets its own collider and material under its own prim tree.
            ground_xf = UsdGeom.Xform.Define(stage, f"{env_prim_path}/ground_patch")
            ground_geom = UsdGeom.Cube.Define(stage, f"{env_prim_path}/ground_patch/geom")
            ground_geom.CreateSizeAttr(1.0)
            ground_geom.CreateDisplayColorAttr([patch_color])
            ground_geom_xf = UsdGeom.Xformable(ground_geom.GetPrim())
            self._get_or_add_xform_op(ground_geom_xf, UsdGeom.XformOp.TypeScale).Set(Gf.Vec3f(patch_x, patch_y, patch_z))
            self._get_or_add_xform_op(UsdGeom.Xformable(ground_xf.GetPrim()), UsdGeom.XformOp.TypeTranslate).Set(
                Gf.Vec3f(0.0, 0.0, -0.5 * patch_z)
            )
            ground_prim = ground_geom.GetPrim()
            UsdPhysics.CollisionAPI.Apply(ground_prim)

            mat_path = f"{env_prim_path}/ground_patch/material"
            material = UsdShade.Material.Define(stage, mat_path)
            material_prim = material.GetPrim()
            material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
            material_api.CreateStaticFrictionAttr(static_friction)
            material_api.CreateDynamicFrictionAttr(dynamic_friction)
            material_api.CreateRestitutionAttr(float(self.cfg.ground_restitution))
            physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
            physx_material_api.CreateFrictionCombineModeAttr("average")
            physx_material_api.CreateRestitutionCombineModeAttr("average")
            UsdShade.MaterialBindingAPI.Apply(ground_prim).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics"
            )
        print(
            "[INFO] Ground friction samples "
            f"static[min={float(self._ground_static_friction.min().item()):.3f}, "
            f"max={float(self._ground_static_friction.max().item()):.3f}] "
            f"dynamic[min={float(self._ground_dynamic_friction.min().item()):.3f}, "
            f"max={float(self._ground_dynamic_friction.max().item()):.3f}]"
        )

    def _setup_env_boundaries(self) -> None:
        stage = get_current_stage()
        # Draw bounds for the actual sampled navigation region.
        half_extent_x, half_extent_y = self._get_nav_half_extents()
        boundary_z = self.cfg.boundary_height
        thickness = self.cfg.boundary_thickness
        color = Gf.Vec3f(*self.cfg.boundary_color)

        edge_specs = (
            ("north", (0.0, half_extent_y, boundary_z), (2.0 * half_extent_x, thickness, thickness)),
            ("south", (0.0, -half_extent_y, boundary_z), (2.0 * half_extent_x, thickness, thickness)),
            ("east", (half_extent_x, 0.0, boundary_z), (thickness, 2.0 * half_extent_y, thickness)),
            ("west", (-half_extent_x, 0.0, boundary_z), (thickness, 2.0 * half_extent_y, thickness)),
        )
        for env_prim_path in self.scene.env_prim_paths:
            for edge_name, pos_xyz, scale_xyz in edge_specs:
                edge_xf_path = f"{env_prim_path}/env_boundary/{edge_name}"
                edge_geom_path = f"{edge_xf_path}/geom"
                edge_xf = UsdGeom.Xform.Define(stage, edge_xf_path)
                edge_cube = UsdGeom.Cube.Define(stage, edge_geom_path)
                edge_cube.CreateSizeAttr(1.0)
                edge_cube.CreateDisplayColorAttr([color])
                edge_geom_xformable = UsdGeom.Xformable(edge_cube.GetPrim())
                edge_xf_xformable = UsdGeom.Xformable(edge_xf.GetPrim())
                self._get_or_add_xform_op(edge_geom_xformable, UsdGeom.XformOp.TypeScale).Set(Gf.Vec3f(*scale_xyz))
                self._get_or_add_xform_op(edge_xf_xformable, UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3f(*pos_xyz))

    def _sample_start_goal(self, env_ids_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs = len(env_ids_t)
        env_origins_xy = self.scene.env_origins[env_ids_t, 0:2]
        half_extent_x, half_extent_y = self._get_nav_half_extents()
        goal_margin = float(self.cfg.goal_border_margin)
        start_margin = float(self.cfg.start_corner_margin)

        # Same corner in every env (south-west corner), slightly inset from border.
        start_local = torch.zeros((num_envs, 2), dtype=torch.float32, device=self.device)
        start_local[:, 0] = -half_extent_x + start_margin
        start_local[:, 1] = -half_extent_y + start_margin
        start_xy = env_origins_xy + start_local

        goal_low_x = -half_extent_x + goal_margin
        goal_high_x = half_extent_x - goal_margin
        goal_low_y = -half_extent_y + goal_margin
        goal_high_y = half_extent_y - goal_margin

        goal_xy = env_origins_xy + torch.stack(
            (
                sample_uniform(goal_low_x, goal_high_x, (num_envs,), self.device),
                sample_uniform(goal_low_y, goal_high_y, (num_envs,), self.device),
            ),
            dim=-1,
        )

        for _ in range(24):
            goal_local = goal_xy - env_origins_xy
            goal_oob = (torch.abs(goal_local[:, 0]) > (half_extent_x - goal_margin)) | (
                torch.abs(goal_local[:, 1]) > (half_extent_y - goal_margin)
            )
            dist = torch.linalg.norm(goal_xy - start_xy, dim=-1)
            invalid = goal_oob | (dist < self.cfg.min_start_goal_distance)
            if not torch.any(invalid):
                break
            resample_count = int(invalid.sum().item())
            invalid_env_ids = env_ids_t[invalid]
            invalid_origins_xy = self.scene.env_origins[invalid_env_ids, 0:2]
            goal_xy[invalid] = invalid_origins_xy + torch.stack(
                (
                    sample_uniform(goal_low_x, goal_high_x, (resample_count,), self.device),
                    sample_uniform(goal_low_y, goal_high_y, (resample_count,), self.device),
                ),
                dim=-1,
            )

        return start_xy, goal_xy

    def _reset_nav_markers(self, env_ids_t: torch.Tensor) -> None:
        for env_id in env_ids_t.tolist():
            sx, sy = self._start_xy_w[env_id].tolist()
            gx, gy = self._goal_xy_w[env_id].tolist()
            # Marker prims live under /World/envs/env_i, so transform must be env-local.
            ox, oy = self.scene.env_origins[env_id, 0:2].tolist()
            self._start_marker_translate_ops[env_id].Set(Gf.Vec3f(sx - ox, sy - oy, 0.04))
            self._goal_marker_translate_ops[env_id].Set(Gf.Vec3f(gx - ox, gy - oy, 0.04))

    @staticmethod
    def _get_or_add_xform_op(xformable: UsdGeom.Xformable, op_type: UsdGeom.XformOp.Type) -> UsdGeom.XformOp:
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == op_type:
                return op
        if op_type == UsdGeom.XformOp.TypeScale:
            return xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat)
        if op_type == UsdGeom.XformOp.TypeTranslate:
            return xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionFloat)
        if op_type == UsdGeom.XformOp.TypeOrient:
            return xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat)
        raise RuntimeError(f"Unsupported xform op type: {op_type}")

    def _get_nav_half_extents(self) -> tuple[float, float]:
        half_extent_x = max(
            abs(float(self.cfg.start_random_x_range[0])),
            abs(float(self.cfg.start_random_x_range[1])),
            abs(float(self.cfg.goal_random_x_range[0])),
            abs(float(self.cfg.goal_random_x_range[1])),
        )
        half_extent_y = max(
            abs(float(self.cfg.start_random_y_range[0])),
            abs(float(self.cfg.start_random_y_range[1])),
            abs(float(self.cfg.goal_random_y_range[0])),
            abs(float(self.cfg.goal_random_y_range[1])),
        )
        return half_extent_x, half_extent_y
