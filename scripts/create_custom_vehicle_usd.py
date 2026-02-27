#!/usr/bin/env python3
"""Create a simple articulated vehicle USD for Isaac Lab.

The generated articulation has:
- one chassis rigid body
- four suspension slider joints (z travel)
- two front steering joints (yaw)
- four wheel spin joints (axle rotation)

Joint naming is aligned with the task config defaults:
- .*front_left_wheel.*
- .*rear_left_wheel.*
- .*front_right_wheel.*
- .*rear_right_wheel.*
- .*front_left_steer.*
- .*front_right_steer.*
- .*suspension.*
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Create a custom articulated vehicle USD.")
parser.add_argument(
    "--output",
    type=Path,
    default=Path("assets/vehicles/custom_vehicle.usd"),
    help="Output USD path.",
)
parser.add_argument("--wheelbase", type=float, default=2.7, help="Distance between front and rear axle centers [m].")
parser.add_argument("--track-width", type=float, default=1.7, help="Distance between left and right wheels [m].")
parser.add_argument("--chassis-height", type=float, default=0.65, help="Chassis link center height [m].")
parser.add_argument("--wheel-center-height", type=float, default=0.40, help="Wheel center height at rest [m].")
parser.add_argument("--wheel-radius", type=float, default=0.33, help="Wheel radius [m].")
parser.add_argument("--wheel-width", type=float, default=0.24, help="Wheel width [m].")
parser.add_argument("--chassis-mass", type=float, default=900.0, help="Chassis mass [kg].")
parser.add_argument("--wheel-mass", type=float, default=18.0, help="Per-wheel mass [kg].")
parser.add_argument("--suspension-link-mass", type=float, default=25.0, help="Per-corner suspension-link mass [kg].")
parser.add_argument("--steer-link-mass", type=float, default=12.0, help="Per-front-corner steer-link mass [kg].")
parser.add_argument(
    "--suspension-lower-limit",
    type=float,
    default=-0.18,
    help="Suspension prismatic lower limit [m].",
)
parser.add_argument(
    "--suspension-upper-limit",
    type=float,
    default=0.08,
    help="Suspension prismatic upper limit [m].",
)
parser.add_argument(
    "--suspension-stiffness",
    type=float,
    default=28000.0,
    help="Suspension spring stiffness (drive stiffness).",
)
parser.add_argument(
    "--suspension-damping",
    type=float,
    default=3500.0,
    help="Suspension damping (drive damping).",
)
parser.add_argument(
    "--steering-max-angle-deg",
    type=float,
    default=35.0,
    help="Steering joint limit magnitude in degrees.",
)
parser.add_argument(
    "--disable-chassis-collision",
    action="store_true",
    default=False,
    help="Do not author a chassis collision shape.",
)
parser.add_argument("--wheel-static-friction", type=float, default=1.2, help="Wheel collider static friction.")
parser.add_argument("--wheel-dynamic-friction", type=float, default=1.0, help="Wheel collider dynamic friction.")
parser.add_argument("--wheel-restitution", type=float, default=0.0, help="Wheel collider restitution.")
parser.add_argument("--chassis-static-friction", type=float, default=0.9, help="Chassis collider static friction.")
parser.add_argument("--chassis-dynamic-friction", type=float, default=0.8, help="Chassis collider dynamic friction.")
parser.add_argument("--chassis-restitution", type=float, default=0.0, help="Chassis collider restitution.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema


@dataclass(frozen=True)
class WheelCorner:
    name: str
    pos_x: float
    pos_y: float
    steering: bool


def _build_wheel_corners(wheelbase: float, track_width: float) -> tuple[WheelCorner, WheelCorner, WheelCorner, WheelCorner]:
    half_wheelbase = 0.5 * wheelbase
    half_track = 0.5 * track_width
    return (
        WheelCorner("front_left", half_wheelbase, half_track, True),
        WheelCorner("front_right", half_wheelbase, -half_track, True),
        WheelCorner("rear_left", -half_wheelbase, half_track, False),
        WheelCorner("rear_right", -half_wheelbase, -half_track, False),
    )


def _apply_rigid_body(prim: Usd.Prim, mass: float) -> None:
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)


def _add_box_visual_and_collider(
    stage: Usd.Stage,
    parent_path: Sdf.Path,
    name: str,
    size_xyz: tuple[float, float, float],
    color: tuple[float, float, float],
    local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    prim_path = parent_path.AppendChild(name)
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(size_xyz[0], size_xyz[1], size_xyz[2]))
    cube.AddTranslateOp().Set(Gf.Vec3f(*local_pos))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _add_box_visual(
    stage: Usd.Stage,
    parent_path: Sdf.Path,
    name: str,
    size_xyz: tuple[float, float, float],
    color: tuple[float, float, float],
    local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    prim_path = parent_path.AppendChild(name)
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(size_xyz[0], size_xyz[1], size_xyz[2]))
    cube.AddTranslateOp().Set(Gf.Vec3f(*local_pos))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _add_box_collision(
    stage: Usd.Stage,
    parent_path: Sdf.Path,
    name: str,
    size_xyz: tuple[float, float, float],
    local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    material_path: Sdf.Path | None = None,
) -> None:
    prim_path = parent_path.AppendChild(name)
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(size_xyz[0], size_xyz[1], size_xyz[2]))
    cube.AddTranslateOp().Set(Gf.Vec3f(*local_pos))
    cube.CreateVisibilityAttr("invisible")
    cube_prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(cube_prim)
    if material_path is not None:
        _bind_physics_material(stage, cube_prim, material_path)


def _add_wheel_visual_and_collider(
    stage: Usd.Stage,
    parent_path: Sdf.Path,
    radius: float,
    width: float,
    color: tuple[float, float, float],
    material_path: Sdf.Path | None = None,
) -> None:
    cyl_path = parent_path.AppendChild("wheel_geom")
    cyl = UsdGeom.Cylinder.Define(stage, cyl_path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(width)
    cyl.CreateAxisAttr("Y")
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    cyl_prim = cyl.GetPrim()
    UsdPhysics.CollisionAPI.Apply(cyl_prim)
    if material_path is not None:
        _bind_physics_material(stage, cyl_prim, material_path)


def _create_physics_material(
    stage: Usd.Stage,
    path: Sdf.Path,
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
) -> Sdf.Path:
    material = UsdShade.Material.Define(stage, path)
    material_prim = material.GetPrim()
    material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    material_api.CreateStaticFrictionAttr(static_friction)
    material_api.CreateDynamicFrictionAttr(dynamic_friction)
    material_api.CreateRestitutionAttr(restitution)
    physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
    physx_material_api.CreateFrictionCombineModeAttr("average")
    physx_material_api.CreateRestitutionCombineModeAttr("average")
    return path


def _bind_physics_material(stage: Usd.Stage, target_prim: Usd.Prim, material_path: Sdf.Path) -> None:
    material = UsdShade.Material.Get(stage, material_path)
    UsdShade.MaterialBindingAPI.Apply(target_prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _define_prismatic_joint(
    stage: Usd.Stage,
    joint_path: Sdf.Path,
    body0: Sdf.Path,
    body1: Sdf.Path,
    local_pos0: tuple[float, float, float],
    local_pos1: tuple[float, float, float],
    lower: float,
    upper: float,
    damping: float,
    stiffness: float,
) -> None:
    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    joint.CreateAxisAttr("Z")
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*local_pos0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*local_pos1))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateDampingAttr(damping)
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)


def _define_revolute_joint(
    stage: Usd.Stage,
    joint_path: Sdf.Path,
    body0: Sdf.Path,
    body1: Sdf.Path,
    axis: str,
    local_pos0: tuple[float, float, float],
    local_pos1: tuple[float, float, float],
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateAxisAttr(axis)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*local_pos0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*local_pos1))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    if lower is not None:
        joint.CreateLowerLimitAttr(lower)
    if upper is not None:
        joint.CreateUpperLimitAttr(upper)


def create_vehicle_usd(output_usd: Path) -> None:
    stage = Usd.Stage.CreateNew(str(output_usd))
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(0)
    stage.SetTimeCodesPerSecond(60)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
    stage.SetDefaultPrim(world.GetPrim())

    vehicle_xf = UsdGeom.Xform.Define(stage, Sdf.Path("/World/Vehicle"))
    vehicle_prim = vehicle_xf.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(vehicle_prim)
    wheel_corners = _build_wheel_corners(args_cli.wheelbase, args_cli.track_width)
    materials_root = Sdf.Path("/World/Vehicle/Materials")
    wheel_material_path = _create_physics_material(
        stage,
        materials_root.AppendChild("WheelMaterial"),
        static_friction=args_cli.wheel_static_friction,
        dynamic_friction=args_cli.wheel_dynamic_friction,
        restitution=args_cli.wheel_restitution,
    )
    chassis_material_path = _create_physics_material(
        stage,
        materials_root.AppendChild("ChassisMaterial"),
        static_friction=args_cli.chassis_static_friction,
        dynamic_friction=args_cli.chassis_dynamic_friction,
        restitution=args_cli.chassis_restitution,
    )

    chassis_path = Sdf.Path("/World/Vehicle/chassis")
    chassis = UsdGeom.Xform.Define(stage, chassis_path)
    chassis.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, args_cli.chassis_height))
    _apply_rigid_body(chassis.GetPrim(), mass=args_cli.chassis_mass)
    _add_box_visual(
        stage,
        chassis_path,
        "chassis_visual",
        size_xyz=(2.6, 1.4, 0.5),
        color=(0.15, 0.45, 0.85),
    )
    # Dedicated chassis collider for reliable body-to-body contacts.
    if not args_cli.disable_chassis_collision:
        _add_box_collision(
            stage,
            chassis_path,
            "chassis_collision",
            size_xyz=(2.65, 1.45, 0.55),
            material_path=chassis_material_path,
        )

    for corner in wheel_corners:
        suspension_link_path = Sdf.Path(f"/World/Vehicle/{corner.name}_suspension_link")
        suspension_link = UsdGeom.Xform.Define(stage, suspension_link_path)
        suspension_link.AddTranslateOp().Set(
            Gf.Vec3f(corner.pos_x, corner.pos_y, args_cli.wheel_center_height + 0.15)
        )
        _apply_rigid_body(suspension_link.GetPrim(), mass=args_cli.suspension_link_mass)
        _add_box_visual(
            stage,
            suspension_link_path,
            "suspension_geom",
            size_xyz=(0.12, 0.12, 0.24),
            color=(0.75, 0.75, 0.75),
        )
        _define_prismatic_joint(
            stage=stage,
            joint_path=Sdf.Path(f"/World/Vehicle/{corner.name}_suspension_joint"),
            body0=chassis_path,
            body1=suspension_link_path,
            local_pos0=(corner.pos_x, corner.pos_y, -0.05),
            local_pos1=(0.0, 0.0, 0.0),
            lower=args_cli.suspension_lower_limit,
            upper=args_cli.suspension_upper_limit,
            damping=args_cli.suspension_damping,
            stiffness=args_cli.suspension_stiffness,
        )

        if corner.steering:
            steer_link_path = Sdf.Path(f"/World/Vehicle/{corner.name}_steer_link")
            steer_link = UsdGeom.Xform.Define(stage, steer_link_path)
            steer_link.AddTranslateOp().Set(Gf.Vec3f(corner.pos_x, corner.pos_y, args_cli.wheel_center_height))
            _apply_rigid_body(steer_link.GetPrim(), mass=args_cli.steer_link_mass)
            _add_box_visual(
                stage,
                steer_link_path,
                "steer_geom",
                size_xyz=(0.10, 0.10, 0.10),
                color=(0.35, 0.35, 0.35),
            )

            _define_revolute_joint(
                stage=stage,
                joint_path=Sdf.Path(f"/World/Vehicle/{corner.name}_steer_joint"),
                body0=suspension_link_path,
                body1=steer_link_path,
                axis="Z",
                local_pos0=(0.0, 0.0, -0.15),
                local_pos1=(0.0, 0.0, 0.0),
                lower=-args_cli.steering_max_angle_deg,
                upper=args_cli.steering_max_angle_deg,
            )
            wheel_parent = steer_link_path
        else:
            wheel_parent = suspension_link_path

        wheel_link_path = Sdf.Path(f"/World/Vehicle/{corner.name}_wheel_link")
        wheel_link = UsdGeom.Xform.Define(stage, wheel_link_path)
        wheel_link.AddTranslateOp().Set(Gf.Vec3f(corner.pos_x, corner.pos_y, args_cli.wheel_center_height))
        _apply_rigid_body(wheel_link.GetPrim(), mass=args_cli.wheel_mass)
        _add_wheel_visual_and_collider(
            stage,
            wheel_link_path,
            radius=args_cli.wheel_radius,
            width=args_cli.wheel_width,
            color=(0.07, 0.07, 0.07),
            material_path=wheel_material_path,
        )

        _define_revolute_joint(
            stage=stage,
            joint_path=Sdf.Path(f"/World/Vehicle/{corner.name}_wheel_joint"),
            body0=wheel_parent,
            body1=wheel_link_path,
            axis="Y",
            local_pos0=(0.0, 0.0, 0.0 if corner.steering else -0.15),
            local_pos1=(0.0, 0.0, 0.0),
        )

    stage.Save()


def main() -> None:
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    create_vehicle_usd(args_cli.output.resolve())
    print(f"Wrote vehicle USD to: {args_cli.output}")


if __name__ == "__main__":
    main()
    simulation_app.close()
