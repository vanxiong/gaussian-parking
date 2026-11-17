"""Use NVIDIA's vehicle authoring API with the CMPE294 parking scene."""
import json
import math
from pathlib import Path
import numpy as np
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
from omni.physxvehicle.scripts.wizards.physxVehicleWizard import VehicleData, create_vehicle
from omni.physxvehicle.scripts.helpers.UnitScale import UnitScale
from omni.physx.scripts.physicsUtils import add_collision_to_collision_group, add_physics_material_to_prim

ROOT = Path(__file__).resolve().parent
CAR_PATH = '/World/PhysXCar'
START = (-5., -2., .67)
TARGET = (-2.8, 6.9, math.pi/2)

def build(out):
    omni.usd.get_context().open_stage(str(ROOT/'output'/'parking_scene.usda'))
    stage = omni.usd.get_context().get_stage()
    layer = stage.GetRootLayer()
    car = UsdGeom.Xform.Define(stage, CAR_PATH)
    car.AddTranslateOp().Set(Gf.Vec3d(*START))
    car.AddOrientOp().Set(Gf.Quatf(1,0,0,0))
    # Keep the original sedan/camera geometry, positioned relative to the chassis center.
    appearance = CAR_PATH+'/Appearance'
    Sdf.CopySpec(layer, '/World/EgoVehicle', layer, appearance)
    stage.GetPrimAtPath(appearance).GetAttribute('xformOp:translate').Set(Gf.Vec3d(0,0,-.62))
    for suffix in ('FL','FR','BL','BR'):
        for part in ('Tire','Rim'):
            stage.RemovePrim(appearance+'/'+part+suffix)
    UsdPhysics.CollisionAPI(stage.GetPrimAtPath(appearance+'/Body')).GetCollisionEnabledAttr().Set(True)

    data = VehicleData(UnitScale(1.,1.), VehicleData.AXIS_Z, VehicleData.AXIS_X)
    data.vehiclePath = CAR_PATH
    data.rootVehiclePath = '/World/SedanVehicleData'
    data.rootSharedPath = '/World/PhysXVehicleData'
    data.chassisLength, data.chassisWidth, data.chassisHeight = 4.3, 1.82, .7
    data.chassisMass = 1400.
    data.tireRadius, data.tireWidth, data.tireMass = .38, .23, 22.
    data.horsepower, data.engineMaxRPM = 150., 6000.
    data.damping[0] = data.damping[1] = .7
    data.maxSteerAngle[0] = math.degrees(.85)
    # Set individual front-wheel angles from the controller's center steering
    # angle, avoiding the native Ackermann API's reference-wheel convention.
    data.useAckermannCorrection = False
    for i, suffix in enumerate(('FL','FR','BL','BR')):
        x = 1.35 if i<2 else -1.35
        y = .94 if i%2==0 else -.94
        wheel_path = CAR_PATH+'/Wheels/'+suffix
        wheel = UsdGeom.Xform.Define(stage, wheel_path)
        wheel.AddTranslateOp().Set(Gf.Vec3d(x,y,-.23))
        tire = data.tireList[i]
        tire.radius, tire.width, tire.mass, tire.path = .38, .23, 22., wheel_path
        for part in ('Tire','Rim'):
            dest = wheel_path+'/'+part
            Sdf.CopySpec(layer, '/World/EgoVehicle/'+part+suffix, layer, dest)
            offset = (0,0,0) if part=='Tire' else (0,.125 if y>0 else -.125,0)
            stage.GetPrimAtPath(dest).GetAttribute('xformOp:translate').Set(Gf.Vec3d(*offset))
    messages, _ = create_vehicle(stage, data)
    if not car.GetPrim().HasAPI(PhysxSchema.PhysxVehicleAPI):
        raise RuntimeError('Vehicle authoring failed: '+str(messages))
    mass = UsdPhysics.MassAPI(car)
    mass.GetCenterOfMassAttr().Set(Gf.Vec3f(0,0,-.07))
    mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(1400*(1.82**2+.7**2)/12,
        1400*(4.3**2+.7**2)/12,1400*(4.3**2+1.82**2)/12))
    # The physics shape now has the same proportions as the sedan body.
    add_collision_to_collision_group(stage, appearance+'/Body',data.rootSharedPath+'/VehicleChassisCollisionGroup')
    for path in ('/World/Lot','/World/Ground'):
        add_collision_to_collision_group(stage,path,data.rootSharedPath+'/GroundSurfaceCollisionGroup')
        add_physics_material_to_prim(stage,stage.GetPrimAtPath(path),Sdf.Path(data.rootSharedPath+'/VehicleGroundMaterial'))
    stage.GetPrimAtPath('/World/EgoVehicle').SetActive(False)
    for i,(pos,size) in enumerate([((-4.12,7.3,.075),(.05,5.1,.012)),
        ((-1.48,7.3,.075),(.05,5.1,.012)),((-2.8,9.85,.075),(2.64,.05,.012))]):
        cube = UsdGeom.Cube.Define(stage,f'/World/ParkingTarget/Edge{i}')
        cube.CreateSizeAttr(1)
        cube.AddTranslateOp().Set(Gf.Vec3d(*pos))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath('/World/Materials/teal')))
    calibration = json.loads((ROOT/'output'/'camera_calibration.json').read_text())
    calibration['ego_reference'] = 'PhysX chassis frame; X forward, Y left, Z up'
    for record in calibration['cameras']:
        record['path'] = appearance+'/Cameras/'+record['name']
        transform = np.array(record['T_parent_from_optical'])
        transform[2,3] -= .62
        record['T_parent_from_optical'] = transform.tolist()
    (out/'camera_calibration.json').write_text(json.dumps(calibration,indent=2))
    # Conservative oriented footprint monitor, matching the earlier parking demo.
    colliders = []
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_])
    for prim in stage.Traverse():
        name = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.CollisionAPI) and not name.startswith(CAR_PATH) and name not in ('/World/Lot','/World/Ground'):
            bounds = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            colliders.append((name,np.array(bounds.GetMin())[:2],np.array(bounds.GetMax())[:2]))
    return stage, car.GetPrim(), colliders, calibration

def overlaps(position, yaw, colliders):
    forward = np.array([math.cos(yaw),math.sin(yaw)])
    lateral = np.array([-math.sin(yaw),math.cos(yaw)])
    for name, lo, hi in colliders:
        delta = (lo+hi)/2-np.asarray(position)[:2]
        other_half = (hi-lo)/2
        if all(abs(delta@axis)<2.2*abs(forward@axis)+1.10*abs(lateral@axis)+other_half@np.abs(axis)
            for axis in (np.array([1.,0]),np.array([0.,1]),forward,lateral)):
            return name
    return None
