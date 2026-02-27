#!/usr/bin/env python3
"""Build a clean, self-contained PhysX vehicle asset USD for Isaac Lab.

This script intentionally does not author a PhysicsScene, lights, cameras, or a ground plane.
It authors a reusable asset rooted at /Vehicle and keeps all relationship targets in-scope.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


AUTOMATIC_GEAR = 255


@dataclass(frozen=True)
class WheelSpec:
    name: str
    index: int
    position: Gf.Vec3f
    suspension_frame_position: Gf.Vec3f
    lateral_stiffness_y: float


def _set_api_schemas(prim: Usd.Prim, schema_tokens: list[str]) -> None:
    token_list = Sdf.TokenListOp()
    token_list.explicitItems = schema_tokens
    prim.SetMetadata("apiSchemas", token_list)


def _create_attr(prim: Usd.Prim, name: str, sdf_type, value) -> None:
    prim.CreateAttribute(name, sdf_type, custom=True).Set(value)


def _create_rel(prim: Usd.Prim, name: str, targets: list[Sdf.Path]) -> None:
    rel = prim.CreateRelationship(name, custom=True)
    for target in targets:
        rel.AddTarget(target)


def _add_collision_group_include(stage: Usd.Stage, group_path: Sdf.Path, collider_path: Sdf.Path) -> None:
    group_prim = stage.GetPrimAtPath(group_path)
    includes_rel = group_prim.GetRelationship("collection:colliders:includes")
    if not includes_rel:
        includes_rel = group_prim.CreateRelationship("collection:colliders:includes", custom=False)
    includes_rel.AddTarget(collider_path)


def _build_materials_and_friction_tables(stage: Usd.Stage, shared_path: Sdf.Path) -> dict[str, Sdf.Path]:
    tarmac_path = shared_path.AppendChild("TarmacMaterial")
    gravel_path = shared_path.AppendChild("GravelMaterial")

    tarmac_prim = UsdShade.Material.Define(stage, tarmac_path).GetPrim()
    UsdPhysics.MaterialAPI.Apply(tarmac_prim)
    _set_api_schemas(tarmac_prim, ["PhysicsMaterialAPI", "PhysxMaterialAPI"])
    _create_attr(tarmac_prim, "physics:staticFriction", Sdf.ValueTypeNames.Float, 0.9)
    _create_attr(tarmac_prim, "physics:dynamicFriction", Sdf.ValueTypeNames.Float, 0.7)
    _create_attr(tarmac_prim, "physics:restitution", Sdf.ValueTypeNames.Float, 0.0)

    gravel_prim = UsdShade.Material.Define(stage, gravel_path).GetPrim()
    UsdPhysics.MaterialAPI.Apply(gravel_prim)
    _set_api_schemas(gravel_prim, ["PhysicsMaterialAPI", "PhysxMaterialAPI"])
    _create_attr(gravel_prim, "physics:staticFriction", Sdf.ValueTypeNames.Float, 0.6)
    _create_attr(gravel_prim, "physics:dynamicFriction", Sdf.ValueTypeNames.Float, 0.6)
    _create_attr(gravel_prim, "physics:restitution", Sdf.ValueTypeNames.Float, 0.0)

    winter_table_path = shared_path.AppendChild("WinterTireFrictionTable")
    winter_prim = stage.DefinePrim(winter_table_path, "PhysxVehicleTireFrictionTable")
    _create_rel(winter_prim, "groundMaterials", [tarmac_path, gravel_path])
    _create_attr(winter_prim, "frictionValues", Sdf.ValueTypeNames.FloatArray, [0.75, 0.6])

    summer_table_path = shared_path.AppendChild("SummerTireFrictionTable")
    summer_prim = stage.DefinePrim(summer_table_path, "PhysxVehicleTireFrictionTable")
    _create_rel(summer_prim, "groundMaterials", [tarmac_path, gravel_path])
    _create_attr(summer_prim, "frictionValues", Sdf.ValueTypeNames.FloatArray, [0.7, 0.6])

    return {
        "tarmac": tarmac_path,
        "gravel": gravel_path,
        "winter_table": winter_table_path,
        "summer_table": summer_table_path,
    }


def _build_collision_groups(stage: Usd.Stage, shared_path: Sdf.Path) -> dict[str, Sdf.Path]:
    chassis_group_path = shared_path.AppendChild("VehicleChassisCollisionGroup")
    wheel_group_path = shared_path.AppendChild("VehicleWheelCollisionGroup")
    ground_query_group_path = shared_path.AppendChild("VehicleGroundQueryGroup")

    chassis_group = UsdPhysics.CollisionGroup.Define(stage, chassis_group_path)
    chassis_group.CreateFilteredGroupsRel().SetTargets([ground_query_group_path])

    wheel_group = UsdPhysics.CollisionGroup.Define(stage, wheel_group_path)
    wheel_group.CreateFilteredGroupsRel().SetTargets([ground_query_group_path])

    ground_query_group = UsdPhysics.CollisionGroup.Define(stage, ground_query_group_path)
    ground_query_group.CreateFilteredGroupsRel().SetTargets([chassis_group_path, wheel_group_path])

    return {
        "chassis_group": chassis_group_path,
        "wheel_group": wheel_group_path,
        "ground_query_group": ground_query_group_path,
    }


def _build_vehicle_root(stage: Usd.Stage, vehicle_path: Sdf.Path) -> Usd.Prim:
    vehicle_xform = UsdGeom.Xform.Define(stage, vehicle_path)
    prim = vehicle_xform.GetPrim()

    _set_api_schemas(
        prim,
        [
            "PhysxVehicleDriveStandardAPI",
            "PhysxVehicleEngineAPI",
            "PhysxVehicleGearsAPI",
            "PhysxVehicleAutoGearBoxAPI",
            "PhysxVehicleClutchAPI",
            "PhysxVehicleBrakesAPI:brakes0",
            "PhysxVehicleBrakesAPI:brakes1",
            "PhysxVehicleSteeringAPI",
            "PhysxVehicleMultiWheelDifferentialAPI",
            "PhysicsRigidBodyAPI",
            "PhysicsMassAPI",
            "PhysxRigidBodyAPI",
            "PhysxVehicleAPI",
            "PhysxVehicleControllerAPI",
        ],
    )

    custom_data = prim.GetCustomData()
    custom_data["physxVehicle"] = {"referenceFrameIsCenterOfMass": False}
    prim.SetCustomData(custom_data)

    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim)

    xformable = UsdGeom.Xformable(prim)
    xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # Mass/inertia model: box approximation around chassis volume.
    mass = 1800.0
    chassis_dims = Gf.Vec3f(4.6, 1.8, 1.0)
    inertia = Gf.Vec3f(
        (chassis_dims[1] ** 2 + chassis_dims[2] ** 2) * mass / 12.0,
        (chassis_dims[0] ** 2 + chassis_dims[2] ** 2) * mass / 12.0,
        (chassis_dims[0] ** 2 + chassis_dims[1] ** 2) * mass / 12.0,
    )

    _create_attr(prim, "physics:mass", Sdf.ValueTypeNames.Float, mass)
    _create_attr(prim, "physics:centerOfMass", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(0.0, 0.0, -0.15))
    _create_attr(prim, "physics:diagonalInertia", Sdf.ValueTypeNames.Float3, inertia)
    _create_attr(prim, "physics:principalAxes", Sdf.ValueTypeNames.Quatf, Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # PhysX vehicles expect rigid-body gravity disabled; suspension/wheel forces handle support.
    _create_attr(prim, "physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool, True)
    _create_attr(prim, "physxRigidBody:sleepThreshold", Sdf.ValueTypeNames.Float, 0.005)
    _create_attr(prim, "physxRigidBody:stabilizationThreshold", Sdf.ValueTypeNames.Float, 0.001)

    _create_attr(prim, "physxVehicle:vehicleEnabled", Sdf.ValueTypeNames.Bool, True)
    _create_attr(prim, "physxVehicle:subStepThresholdLongitudinalSpeed", Sdf.ValueTypeNames.Float, 5.0)
    _create_attr(prim, "physxVehicle:lowForwardSpeedSubStepCount", Sdf.ValueTypeNames.Int, 3)
    _create_attr(prim, "physxVehicle:highForwardSpeedSubStepCount", Sdf.ValueTypeNames.Int, 1)
    _create_attr(prim, "physxVehicle:minPassiveLongitudinalSlipDenominator", Sdf.ValueTypeNames.Float, 4.0)
    _create_attr(prim, "physxVehicle:minActiveLongitudinalSlipDenominator", Sdf.ValueTypeNames.Float, 0.1)
    _create_attr(prim, "physxVehicle:minLateralSlipDenominator", Sdf.ValueTypeNames.Float, 1.0)
    _create_attr(prim, "physxVehicle:suspensionLineQueryType", Sdf.ValueTypeNames.Token, "raycast")

    _create_attr(prim, "physxVehicleBrakes:brakes0:maxBrakeTorque", Sdf.ValueTypeNames.Float, 3600.0)
    _create_attr(prim, "physxVehicleBrakes:brakes1:wheels", Sdf.ValueTypeNames.IntArray, [2, 3])
    _create_attr(prim, "physxVehicleBrakes:brakes1:maxBrakeTorque", Sdf.ValueTypeNames.Float, 3000.0)

    _create_attr(prim, "physxVehicleSteering:wheels", Sdf.ValueTypeNames.IntArray, [0, 1])
    _create_attr(prim, "physxVehicleSteering:angleMultipliers", Sdf.ValueTypeNames.FloatArray, [1.0, 1.0])
    _create_attr(prim, "physxVehicleSteering:maxSteerAngle", Sdf.ValueTypeNames.Float, 0.554264)

    _create_attr(prim, "physxVehicleMultiWheelDifferential:wheels", Sdf.ValueTypeNames.IntArray, [0, 1])
    _create_attr(prim, "physxVehicleMultiWheelDifferential:torqueRatios", Sdf.ValueTypeNames.FloatArray, [0.5, 0.5])
    _create_attr(
        prim,
        "physxVehicleMultiWheelDifferential:averageWheelSpeedRatios",
        Sdf.ValueTypeNames.FloatArray,
        [0.5, 0.5],
    )

    _create_attr(prim, "physxVehicleEngine:moi", Sdf.ValueTypeNames.Float, 1.0)
    _create_attr(prim, "physxVehicleEngine:peakTorque", Sdf.ValueTypeNames.Float, 1000.0)
    _create_attr(prim, "physxVehicleEngine:maxRotationSpeed", Sdf.ValueTypeNames.Float, 600.0)
    _create_attr(
        prim,
        "physxVehicleEngine:torqueCurve",
        Sdf.ValueTypeNames.Float2Array,
        [Gf.Vec2f(0.0, 0.8), Gf.Vec2f(0.33, 1.0), Gf.Vec2f(1.0, 0.8)],
    )
    _create_attr(prim, "physxVehicleEngine:dampingRateFullThrottle", Sdf.ValueTypeNames.Float, 0.15)
    _create_attr(prim, "physxVehicleEngine:dampingRateZeroThrottleClutchEngaged", Sdf.ValueTypeNames.Float, 2.0)
    _create_attr(prim, "physxVehicleEngine:dampingRateZeroThrottleClutchDisengaged", Sdf.ValueTypeNames.Float, 0.35)

    _create_attr(prim, "physxVehicleGears:ratios", Sdf.ValueTypeNames.FloatArray, [-4.0, 4.0, 2.0, 1.5, 1.1, 1.0])
    _create_attr(prim, "physxVehicleGears:ratioScale", Sdf.ValueTypeNames.Float, 4.0)
    _create_attr(prim, "physxVehicleGears:switchTime", Sdf.ValueTypeNames.Float, 0.5)

    _create_attr(prim, "physxVehicleAutoGearBox:upRatios", Sdf.ValueTypeNames.FloatArray, [0.65, 0.65, 0.65, 0.65])
    _create_attr(prim, "physxVehicleAutoGearBox:downRatios", Sdf.ValueTypeNames.FloatArray, [0.5, 0.5, 0.5, 0.5])
    _create_attr(prim, "physxVehicleAutoGearBox:latency", Sdf.ValueTypeNames.Float, 2.0)

    _create_attr(prim, "physxVehicleClutch:strength", Sdf.ValueTypeNames.Float, 10.0)

    _create_attr(prim, "physxVehicleController:accelerator", Sdf.ValueTypeNames.Float, 0.0)
    _create_attr(prim, "physxVehicleController:brake0", Sdf.ValueTypeNames.Float, 0.0)
    _create_attr(prim, "physxVehicleController:brake1", Sdf.ValueTypeNames.Float, 0.0)
    _create_attr(prim, "physxVehicleController:steer", Sdf.ValueTypeNames.Float, 0.0)
    _create_attr(prim, "physxVehicleController:targetGear", Sdf.ValueTypeNames.Int, AUTOMATIC_GEAR)

    return prim


def _build_chassis(stage: Usd.Stage, vehicle_path: Sdf.Path, collision_group_path: Sdf.Path) -> None:
    chassis_collision_path = vehicle_path.AppendChild("ChassisCollision")
    chassis_collision = UsdGeom.Cube.Define(stage, chassis_collision_path)
    chassis_collision.CreatePurposeAttr(UsdGeom.Tokens.guide)
    chassis_collision.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(2.3, 0.9, 0.5))
    chassis_collision.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(0.0, 0.0, 0.05))

    chassis_collision_prim = chassis_collision.GetPrim()
    UsdPhysics.CollisionAPI.Apply(chassis_collision_prim)
    _set_api_schemas(chassis_collision_prim, ["PhysicsCollisionAPI", "PhysxCollisionAPI"])
    _create_attr(chassis_collision_prim, "physxCollision:restOffset", Sdf.ValueTypeNames.Float, 0.0)
    _create_attr(chassis_collision_prim, "physxCollision:contactOffset", Sdf.ValueTypeNames.Float, 0.02)
    _add_collision_group_include(stage, collision_group_path, chassis_collision_path)

    chassis_render_path = vehicle_path.AppendChild("ChassisRender")
    chassis_render = UsdGeom.Mesh.Define(stage, chassis_render_path)
    chassis_render.CreateDisplayColorAttr([Gf.Vec3f(0.20, 0.50, 0.90)])

    hx, hy, hz = 2.1, 0.75, 0.45
    points = [
        Gf.Vec3f(-hx, -hy, -hz),
        Gf.Vec3f(hx, -hy, -hz),
        Gf.Vec3f(hx, hy, -hz),
        Gf.Vec3f(-hx, hy, -hz),
        Gf.Vec3f(-hx, -hy, hz),
        Gf.Vec3f(hx, -hy, hz),
        Gf.Vec3f(hx, hy, hz),
        Gf.Vec3f(-hx, hy, hz),
    ]
    chassis_render.CreatePointsAttr(points)
    chassis_render.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
    chassis_render.CreateFaceVertexIndicesAttr(
        [
            0,
            1,
            2,
            3,
            4,
            7,
            6,
            5,
            0,
            4,
            5,
            1,
            1,
            5,
            6,
            2,
            2,
            6,
            7,
            3,
            3,
            7,
            4,
            0,
        ]
    )


def _build_wheel(
    stage: Usd.Stage,
    vehicle_path: Sdf.Path,
    wheel_spec: WheelSpec,
    wheel_radius: float,
    wheel_width: float,
    tire_long_stiffness: float,
    spring_strength: float,
    spring_damping: float,
    travel_distance: float,
    friction_table_path: Sdf.Path,
    ground_query_group_path: Sdf.Path,
    wheel_collision_group_path: Sdf.Path,
) -> None:
    wheel_path = vehicle_path.AppendChild(wheel_spec.name)
    wheel_xform = UsdGeom.Xform.Define(stage, wheel_path)
    wheel_prim = wheel_xform.GetPrim()

    _set_api_schemas(
        wheel_prim,
        ["PhysxVehicleWheelAttachmentAPI", "PhysxVehicleSuspensionAPI", "PhysxVehicleTireAPI", "PhysxVehicleWheelAPI"],
    )

    wheel_xformable = UsdGeom.Xformable(wheel_prim)
    wheel_xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(wheel_spec.position)

    _create_rel(wheel_prim, "physxVehicleWheelAttachment:collisionGroup", [ground_query_group_path])
    _create_attr(wheel_prim, "physxVehicleWheelAttachment:index", Sdf.ValueTypeNames.Int, wheel_spec.index)
    _create_attr(
        wheel_prim,
        "physxVehicleWheelAttachment:suspensionTravelDirection",
        Sdf.ValueTypeNames.Vector3f,
        Gf.Vec3f(0.0, 0.0, -1.0),
    )
    _create_attr(
        wheel_prim,
        "physxVehicleWheelAttachment:suspensionFramePosition",
        Sdf.ValueTypeNames.Point3f,
        wheel_spec.suspension_frame_position,
    )

    _create_attr(wheel_prim, "physxVehicleWheel:radius", Sdf.ValueTypeNames.Float, wheel_radius)
    _create_attr(wheel_prim, "physxVehicleWheel:width", Sdf.ValueTypeNames.Float, wheel_width)
    _create_attr(wheel_prim, "physxVehicleWheel:mass", Sdf.ValueTypeNames.Float, 20.0)
    _create_attr(wheel_prim, "physxVehicleWheel:moi", Sdf.ValueTypeNames.Float, 1.225)
    _create_attr(wheel_prim, "physxVehicleWheel:dampingRate", Sdf.ValueTypeNames.Float, 0.25)

    _create_attr(
        wheel_prim,
        "physxVehicleTire:lateralStiffnessGraph",
        Sdf.ValueTypeNames.Float2,
        Gf.Vec2f(2.0, wheel_spec.lateral_stiffness_y),
    )
    _create_attr(wheel_prim, "physxVehicleTire:longitudinalStiffness", Sdf.ValueTypeNames.Float, tire_long_stiffness)
    _create_attr(wheel_prim, "physxVehicleTire:camberStiffness", Sdf.ValueTypeNames.Float, 0.0)
    _create_attr(
        wheel_prim,
        "physxVehicleTire:frictionVsSlipGraph",
        Sdf.ValueTypeNames.Float2Array,
        [Gf.Vec2f(0.0, 1.0), Gf.Vec2f(0.1, 1.0), Gf.Vec2f(1.0, 1.0)],
    )
    _create_rel(wheel_prim, "physxVehicleTire:frictionTable", [friction_table_path])

    _create_attr(wheel_prim, "physxVehicleSuspension:springStrength", Sdf.ValueTypeNames.Float, spring_strength)
    _create_attr(wheel_prim, "physxVehicleSuspension:springDamperRate", Sdf.ValueTypeNames.Float, spring_damping)
    _create_attr(wheel_prim, "physxVehicleSuspension:travelDistance", Sdf.ValueTypeNames.Float, travel_distance)

    wheel_collision_path = wheel_path.AppendChild("Collision")
    wheel_collision = UsdGeom.Cylinder.Define(stage, wheel_collision_path)
    wheel_collision.CreatePurposeAttr(UsdGeom.Tokens.guide)
    wheel_collision.CreateAxisAttr(UsdGeom.Tokens.y)
    wheel_collision.CreateHeightAttr(wheel_width)
    wheel_collision.CreateRadiusAttr(wheel_radius)
    wheel_collision.CreateExtentAttr(UsdGeom.Cylinder.ComputeExtentFromPlugins(wheel_collision, 0.0))

    wheel_collision_prim = wheel_collision.GetPrim()
    UsdPhysics.CollisionAPI.Apply(wheel_collision_prim)
    _set_api_schemas(wheel_collision_prim, ["PhysicsCollisionAPI", "PhysxCollisionAPI"])
    _create_attr(wheel_collision_prim, "physxCollision:restOffset", Sdf.ValueTypeNames.Float, 0.0)
    _create_attr(wheel_collision_prim, "physxCollision:contactOffset", Sdf.ValueTypeNames.Float, 0.02)
    _add_collision_group_include(stage, wheel_collision_group_path, wheel_collision_path)

    wheel_render_path = wheel_path.AppendChild("Render")
    wheel_render = UsdGeom.Cylinder.Define(stage, wheel_render_path)
    wheel_render.CreateAxisAttr(UsdGeom.Tokens.y)
    wheel_render.CreateHeightAttr(wheel_width)
    wheel_render.CreateRadiusAttr(wheel_radius)
    wheel_render.CreateExtentAttr(UsdGeom.Cylinder.ComputeExtentFromPlugins(wheel_render, 0.0))


def build_vehicle_asset(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    vehicle_path = Sdf.Path("/Vehicle")
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, vehicle_path).GetPrim())

    shared_path = vehicle_path.AppendChild("SharedVehicleData")
    UsdGeom.Scope.Define(stage, shared_path)

    _build_materials_and_friction_tables(stage, shared_path)
    groups = _build_collision_groups(stage, shared_path)

    _build_vehicle_root(stage, vehicle_path)
    _build_chassis(stage, vehicle_path, groups["chassis_group"])

    wheel_radius = 0.35
    wheel_width = 0.15
    spring_strength = 45000.0
    spring_damping = 4500.0
    travel_distance = 0.18
    tire_long_stiffness = 5000.0

    wheel_specs = [
        WheelSpec(
            name="FrontLeftWheel",
            index=0,
            position=Gf.Vec3f(1.55, 0.82, -0.52),
            suspension_frame_position=Gf.Vec3f(1.55, 0.82, -0.25),
            lateral_stiffness_y=76000.0,
        ),
        WheelSpec(
            name="FrontRightWheel",
            index=1,
            position=Gf.Vec3f(1.55, -0.82, -0.52),
            suspension_frame_position=Gf.Vec3f(1.55, -0.82, -0.25),
            lateral_stiffness_y=76000.0,
        ),
        WheelSpec(
            name="RearLeftWheel",
            index=2,
            position=Gf.Vec3f(-1.55, 0.82, -0.52),
            suspension_frame_position=Gf.Vec3f(-1.55, 0.82, -0.25),
            lateral_stiffness_y=111000.0,
        ),
        WheelSpec(
            name="RearRightWheel",
            index=3,
            position=Gf.Vec3f(-1.55, -0.82, -0.52),
            suspension_frame_position=Gf.Vec3f(-1.55, -0.82, -0.25),
            lateral_stiffness_y=111000.0,
        ),
    ]

    for wheel_spec in wheel_specs:
        _build_wheel(
            stage=stage,
            vehicle_path=vehicle_path,
            wheel_spec=wheel_spec,
            wheel_radius=wheel_radius,
            wheel_width=wheel_width,
            tire_long_stiffness=tire_long_stiffness,
            spring_strength=spring_strength,
            spring_damping=spring_damping,
            travel_distance=travel_distance,
            friction_table_path=shared_path.AppendChild("WinterTireFrictionTable"),
            ground_query_group_path=groups["ground_query_group"],
            wheel_collision_group_path=groups["wheel_group"],
        )

    stage.GetRootLayer().Save()
    return output_path


def _prim_tree(stage: Usd.Stage, root_path: Sdf.Path) -> list[str]:
    lines: list[str] = []

    def _walk(path: Sdf.Path, indent: int) -> None:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return
        lines.append(f"{'  ' * indent}{path.pathString}")
        children = sorted(prim.GetChildren(), key=lambda p: p.GetPath().pathString)
        for child in children:
            _walk(child.GetPath(), indent + 1)

    _walk(root_path, 0)
    return lines


def validate_asset(asset_path: Path) -> dict[str, object]:
    stage = Usd.Stage.Open(str(asset_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD: {asset_path}")

    vehicle_path = Sdf.Path("/Vehicle")
    vehicle_prim = stage.GetPrimAtPath(vehicle_path)

    out_of_scope_targets: list[str] = []
    for prim in Usd.PrimRange(vehicle_prim):
        for rel in prim.GetRelationships():
            for target in rel.GetTargets():
                if not target.pathString.startswith("/Vehicle"):
                    out_of_scope_targets.append(f"{prim.GetPath()} :: {rel.GetName()} -> {target}")

    api_locations: dict[str, list[str]] = {"PhysxVehicleAPI": [], "PhysxVehicleControllerAPI": []}
    for prim in Usd.PrimRange(vehicle_prim):
        api_schemas = prim.GetMetadata("apiSchemas")
        if not api_schemas:
            continue
        tokens = set(getattr(api_schemas, "GetAppliedItems", lambda: [])())
        if "PhysxVehicleAPI" in tokens:
            api_locations["PhysxVehicleAPI"].append(prim.GetPath().pathString)
        if "PhysxVehicleControllerAPI" in tokens:
            api_locations["PhysxVehicleControllerAPI"].append(prim.GetPath().pathString)

    controller_attrs = {
        "accelerator": vehicle_prim.GetAttribute("physxVehicleController:accelerator").Get(),
        "brake0": vehicle_prim.GetAttribute("physxVehicleController:brake0").Get(),
        "brake1": vehicle_prim.GetAttribute("physxVehicleController:brake1").Get(),
        "steer": vehicle_prim.GetAttribute("physxVehicleController:steer").Get(),
        "targetGear": vehicle_prim.GetAttribute("physxVehicleController:targetGear").Get(),
    }

    has_physics_scene = any(p.GetTypeName() == "PhysicsScene" for p in stage.Traverse())

    return {
        "asset_path": str(asset_path),
        "default_prim": stage.GetDefaultPrim().GetPath().pathString if stage.GetDefaultPrim() else None,
        "prim_tree": _prim_tree(stage, vehicle_path),
        "api_locations": api_locations,
        "controller_attrs": controller_attrs,
        "out_of_scope_relationship_targets": out_of_scope_targets,
        "has_physics_scene": has_physics_scene,
    }


def _print_report(result: dict[str, object]) -> None:
    print("=== Vehicle Asset Validation ===")
    print(f"Asset: {result['asset_path']}")
    print(f"Default prim: {result['default_prim']}")
    print(f"Has PhysicsScene prim: {result['has_physics_scene']}")

    print("\nPrim tree under /Vehicle:")
    for line in result["prim_tree"]:
        print(f"  {line}")

    print("\nAPI placements:")
    for api_name, prims in result["api_locations"].items():
        print(f"  {api_name}: {prims}")

    print("\nController attrs on /Vehicle:")
    for name, value in result["controller_attrs"].items():
        print(f"  {name}: {value}")

    out_of_scope = result["out_of_scope_relationship_targets"]
    if out_of_scope:
        print("\nOut-of-scope relationship targets found:")
        for item in out_of_scope:
            print(f"  {item}")
    else:
        print("\nOut-of-scope relationship targets found: none")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean, self-contained PhysX vehicle USD asset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/vehicles/Vehicle_clean.usd"),
        help="Output USD path.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate an existing USD file.",
    )

    args = parser.parse_args()

    if args.validate_only:
        if not args.output.is_file():
            raise FileNotFoundError(f"Asset not found for validation: {args.output}")
        result = validate_asset(args.output)
    else:
        output = build_vehicle_asset(args.output)
        result = validate_asset(output)

    _print_report(result)


if __name__ == "__main__":
    main()
