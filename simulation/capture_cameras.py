"""Capture a stationary, synchronized six-camera input for occupancy development."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gui', action='store_true')
parser.add_argument('--lidar', action='store_true', help='Capture a native PhysX ray-cast LiDAR and ten prior sweeps')
args, _ = parser.parse_known_args()
from isaacsim import SimulationApp
app = SimulationApp({'headless':not args.gui,'width':1600,'height':1000})
from isaacsim.core.utils.extensions import enable_extension
enable_extension('omni.physx.vehicle')
if args.lidar:
    enable_extension('isaacsim.sensors.physx')
app.update()
from isaacsim.core.api import World
import omni.replicator.core as rep
from omni.kit.viewport.utility import get_active_viewport
from pxr import UsdGeom, PhysxSchema
from PIL import Image
from physx_sedan_scene import build

root = Path(__file__).resolve().parent
out = root/'data'/'captures'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
out.mkdir(parents=True)
try:
    stage, car, _, calibration = build(out)
    controller = PhysxSchema.PhysxVehicleControllerAPI(car)
    controller.GetAcceleratorAttr().Set(0.)
    controller.GetBrake0Attr().Set(1.)
    world = World(physics_dt=1/120,rendering_dt=1/120,stage_units_in_meters=1,
                  physics_prim_path='/World/Physics')
    lidar = None
    if args.lidar:
        from isaacsim.sensors.physx import RotatingLidarPhysX
        lidar = world.scene.add(RotatingLidarPhysX(
            prim_path=str(car.GetPath())+'/Lidar', name='occupancy_lidar',
            translation=np.array([0.,0.,1.25]), rotation_frequency=0.,
            fov=(360.,40.), resolution=(0.35,1.25), valid_range=(0.5,70.)))
    get_active_viewport().set_active_camera('/World/Overview')
    world.reset()
    if lidar is not None:
        lidar.add_point_cloud_data_to_frame()
        lidar.add_intensity_data_to_frame()
    for _ in range(240):
        world.step(render=True)
    scans = []
    if lidar is not None:
        for index in range(11):
            for _ in range(12):
                world.step(render=True)
            frame = lidar.get_current_frame()
            xyz = np.asarray(frame['point_cloud']).reshape(-1,3).copy()
            strength = np.asarray(frame['intensity']).reshape(-1).copy()
            if len(xyz) != len(strength):
                raise RuntimeError('LiDAR point/intensity lengths disagree')
            keep = np.isfinite(xyz).all(1) & (strength > 0) & (np.linalg.norm(xyz,axis=1) < 69.9)
            xyz, strength = xyz[keep], strength[keep]
            if not np.isfinite(strength).all() or (strength<0).any() or (strength>255).any():
                raise RuntimeError('Native LiDAR intensity is outside the expected 0..255 range')
            if len(xyz) < 100:
                raise RuntimeError('LiDAR returned too few valid hits')
            world_from_sensor = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(
                stage.GetPrimAtPath(lidar.prim_path))).T
            scans.append((world.current_time, world_from_sensor, xyz, strength))
    world.pause()
    cache = UsdGeom.XformCache()
    world_from_ego = np.array(cache.GetLocalToWorldTransform(car)).T
    lidar_metadata = None
    if scans:
        current_time, world_from_lidar, _, _ = scans[-1]
        points = []
        for timestamp, world_from_sensor, xyz, strength in scans:
            lidar_from_sensor = np.linalg.inv(world_from_lidar) @ world_from_sensor
            aligned = xyz @ lidar_from_sensor[:3,:3].T + lidar_from_sensor[:3,3]
            points.append(np.c_[aligned, strength, np.full(len(xyz),current_time-timestamp)])
        points = np.concatenate(points).astype(np.float32)
        np.save(out/'lidar.npy',points)
        np.save(out/'lidar_current.npy',points[np.isclose(points[:,4],0)])
        lidar_metadata = {'points':'lidar.npy','fields':['x','y','z','intensity','time_lag_seconds'],
            'ego_from_lidar':(np.linalg.inv(world_from_ego)@world_from_lidar).tolist(),
            'simulation_timestamp_seconds':current_time,'sweeps':len(scans),
            'point_count':len(points),'current_point_count':len(scans[-1][2]),
            'sensor':'Isaac Sim RotatingLidarPhysX; 360 degree ray-cast LiDAR',
            'intensity_model':'Native PhysX 0..255 hit indicator; not calibrated material reflectance',
            'scan_model':'instantaneous full scans at 10 Hz; no intra-scan motion distortion',
            'horizontal_resolution_degrees':0.35,'vertical_resolution_degrees':1.25,
            'vertical_fov_degrees':40.,'max_range_m':70.,
            'sweep_registration':'Simulator poses used only to register static capture sweeps; not estimated localization'}
    products = []
    cameras = []
    for record in calibration['cameras']:
        product = rep.create.render_product(record['path'],tuple(record['resolution']))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb')
        rgb.attach([product])
        products.append((record['name'],rgb,product))
        world_from_optical = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(record['path']))).T @ np.diag([1,-1,-1,1])
        ego_from_optical = np.linalg.inv(world_from_ego) @ world_from_optical
        optical_from_ego = np.linalg.inv(ego_from_optical)
        intrinsic = np.array(record['K'])
        cameras.append({'name':record['name'],'image':record['name']+'.png',
            'resolution':record['resolution'],'intrinsic':intrinsic.tolist(),
            'ego_from_optical':ego_from_optical.tolist(),
            'optical_from_ego':optical_from_ego.tolist(),
            'projection_from_ego':(intrinsic @ optical_from_ego[:3,:]).tolist()})
    rep.orchestrator.step(rt_subframes=8,pause_timeline=True)
    checks = {}
    for name,rgb,product in products:
        pixels = np.asarray(rgb.get_data())
        if pixels.size==0:
            raise RuntimeError('Empty frame: '+name)
        pixels = pixels[:,:,:3].astype(np.uint8)
        Image.fromarray(pixels).save(out/(name+'.png'))
        checks[name] = {'width':pixels.shape[1],'height':pixels.shape[0],'std':float(pixels.std())}
        if pixels.std()<2:
            raise RuntimeError('Blank frame: '+name)
    after = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(car)).T
    if not np.allclose(after,world_from_ego,atol=1e-7,rtol=0):
        raise RuntimeError('Car moved during synchronized capture')
    manifest = {'schema':'cmpe294.surround_camera_capture.v1','capture_id':out.name,
        'simulation_timestamp_seconds':world.current_time,'reference_frame':'ego_chassis',
        'reference_axes':'X forward, Y left, Z up','optical_axes':'X right, Y down, Z forward',
        'calibration_status':'Approximate starter camera rig; not nuScenes calibration',
        'world_from_ego':world_from_ego.tolist(),'cameras':cameras,
        'synchronization':'Physics paused; all six cameras rendered in one orchestrator step',
        'validation':checks}
    if lidar_metadata is not None:
        manifest['lidar'] = lidar_metadata
        manifest['synchronization'] = 'Current LiDAR at final physics step; paused RGB at same pose/time. Ten earlier LiDAR sweeps motion-compensated into current LiDAR frame.'
    (out/'capture.json').write_text(json.dumps(manifest,indent=2))
    stage.GetRootLayer().Export(str(out/'scene.usda'))
    (root/'data'/'latest_capture.json').write_text(json.dumps({'capture':out.relative_to(root).as_posix()},indent=2))
    print('CAMERA_CAPTURE_READY',str(out),flush=True)
    if args.gui:
        while app.is_running():
            world.render()
finally:
    app.close()
