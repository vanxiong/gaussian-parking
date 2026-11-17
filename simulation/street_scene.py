"""Street approach variant of the existing parking lot."""
import math
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, Sdf, Usd
import numpy as np
from omni.physx.scripts.physicsUtils import add_collision_to_collision_group, add_physics_material_to_prim
from physx_sedan_scene import build as build_parking, overlaps

START=(-23.,-20.,.67)
INITIAL_YAW=math.pi/2
TARGET=(-2.8,6.9,math.pi/2)

def build(out):
    stage,car,colliders,calibration=build_parking(out)
    stage.RemovePrim('/World/Curbs/XN21')
    colliders[:]=[c for c in colliders if c[0]!='/World/Curbs/XN21']
    def box(name,pos,size,material,collision=False,surface=False):
        path='/World/Street/'+name
        cube=UsdGeom.Cube.Define(stage,path);cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*pos));cube.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath('/World/Materials/'+material)))
        if collision:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            if surface:
                shared='/World/PhysXVehicleData'
                add_collision_to_collision_group(stage,path,shared+'/GroundSurfaceCollisionGroup')
                add_physics_material_to_prim(stage,cube.GetPrim(),Sdf.Path(shared+'/VehicleGroundMaterial'))
            else:
                center=np.array(pos[:2]);half=np.array(size[:2])/2
                colliders.append((path,center-half,center+half))
    box('Road',(-25.5,-5.,-.025),(9.,48.,.15),'asphalt',True,True)
    box('Entrance',(-20.,-3.5,-.025),(6.,15.,.15),'asphalt',True,True)
    box('OuterSidewalk',(-31.,-5.,.18),(2.,48.,.36),'concrete',True)
    box('LotBoundarySouth',(-21.,-13.,.22),(.35,4.,.44),'concrete',True)
    box('LotBoundaryNorth',(-21.,9.5,.22),(.35,11.,.44),'concrete',True)
    for i,y in enumerate(range(-27,19,4)):
        box('CenterDash'+str(i),(-25.5,y,.058),(.12,2.,.012),'yellow')
    box('LaneEdge',(-29.7,-5.,.058),(.10,47.,.012),'white')
    for i,y in enumerate((-11.,4.)):
        box('EntryEdge'+str(i),(-19.2,y,.058),(3.6,.10,.012),'white')
    # An overview that includes both the road and the original parking bays.
    camera=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Overview'))
    camera.ClearXformOpOrder()
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(32,-47,48),Gf.Vec3d(-8,-3,0),Gf.Vec3d(0,0,1)).GetInverse())
    return stage,car,colliders,calibration
