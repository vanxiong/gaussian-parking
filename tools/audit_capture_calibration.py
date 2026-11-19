"""Independently compare saved metadata with camera/lidar USD authored state."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('--output',type=Path,required=True);a,_=p.parse_known_args()
from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
try:
    from pxr import Usd,UsdGeom
    stage=Usd.Stage.Open(str(a.capture/'scene.usda'));cache=UsdGeom.XformCache()
    meta=json.loads((a.capture/'capture.json').read_text());ego=np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/PhysXCar'))).T
    result={'world_from_ego_max_error':float(np.abs(ego-np.asarray(meta['world_from_ego'])).max()),'cameras':[]}
    for c in meta['cameras']:
        prim=stage.GetPrimAtPath('/World/PhysXCar/Appearance/Cameras/'+c['name']);cam=UsdGeom.Camera(prim)
        optical=np.linalg.inv(ego)@np.array(cache.GetLocalToWorldTransform(prim)).T@np.diag([1,-1,-1,1])
        w,h=c['resolution'];f=cam.GetFocalLengthAttr().Get();ha=cam.GetHorizontalApertureAttr().Get();va=cam.GetVerticalApertureAttr().Get()
        k=np.array([[w*f/ha,0,w/2],[0,h*f/va,h/2],[0,0,1]])
        result['cameras'].append({'camera':c['name'],'extrinsic_max_error':float(np.abs(optical-c['ego_from_optical']).max()),'intrinsic_max_pixel_error':float(np.abs(k-c['intrinsic']).max()),'usd_intrinsic':k.tolist(),'aperture_offsets':[cam.GetHorizontalApertureOffsetAttr().Get(),cam.GetVerticalApertureOffsetAttr().Get()]})
    lidar=np.linalg.inv(ego)@np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/PhysXCar/Lidar'))).T
    result['lidar_extrinsic_max_error']=float(np.abs(lidar-meta['lidar']['ego_from_lidar']).max())
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
finally:app.close()
