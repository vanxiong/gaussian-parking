"""Native PhysX LiDAR capture and motion-compensated temporal sweep storage."""
from collections import deque
import numpy as np
from pxr import UsdGeom


class LidarHistory:
    def __init__(self,world,stage,car):
        from isaacsim.sensors.physx import RotatingLidarPhysX
        self.world,self.stage,self.car=world,stage,car
        self.sensor=world.scene.add(RotatingLidarPhysX(
            prim_path=str(car.GetPath())+'/Lidar',name='driving_lidar',translation=np.array([0.,0.,1.25]),
            rotation_frequency=0.,fov=(360.,40.),resolution=(.35,1.25),valid_range=(.5,70.)))
        self.scans=deque(maxlen=11)
        self.last_time=-100.

    def enable(self):
        self.sensor.add_point_cloud_data_to_frame()
        self.sensor.add_intensity_data_to_frame()

    def reset(self):
        self.scans.clear();self.last_time=-100.

    def sample(self,force=False):
        now=float(self.world.current_time)
        if not force and now-self.last_time<.099:return
        frame=self.sensor.get_current_frame()
        if frame.get('point_cloud') is None or frame.get('intensity') is None:return
        xyz=np.asarray(frame['point_cloud']).reshape(-1,3).copy()
        intensity=np.asarray(frame['intensity']).reshape(-1).copy().astype(np.float32)
        if len(xyz)!=len(intensity):raise RuntimeError('LiDAR point/intensity length mismatch')
        good=np.isfinite(xyz).all(1)&np.isfinite(intensity)&(intensity>0)&(np.linalg.norm(xyz,axis=1)<69.9)
        xyz,intensity=xyz[good],intensity[good]
        if len(xyz)<100:return
        if (intensity>255).any():raise RuntimeError('Native LiDAR intensity exceeds 255')
        transform=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(
            self.stage.GetPrimAtPath(self.sensor.prim_path))).T
        if self.scans and abs(now-self.scans[-1][0])<1e-6:self.scans.pop()
        self.scans.append((now,transform,xyz,intensity));self.last_time=now

    def snapshot(self,ego):
        self.sample(force=True)
        if not self.scans:raise RuntimeError('LiDAR has not produced a valid scan')
        now,reference,_,_=self.scans[-1]
        if abs(now-float(self.world.current_time))>1e-4:raise RuntimeError('LiDAR scan is stale at capture')
        arrays=[]
        for timestamp,transform,xyz,intensity in self.scans:
            if timestamp>now+1e-6:raise RuntimeError('LiDAR sweep timestamp is in the future')
            aligned=np.linalg.inv(reference)@transform
            points=xyz@aligned[:3,:3].T+aligned[:3,3]
            arrays.append(np.c_[points,intensity,np.full(len(points),max(0.,now-timestamp))])
        points=np.concatenate(arrays).astype(np.float32)
        metadata={'points':'lidar.npy','fields':['x','y','z','intensity','time_lag_seconds'],
                  'ego_from_lidar':(np.linalg.inv(ego)@reference).tolist(),
                  'simulation_timestamp_seconds':now,'sweeps':len(arrays),'point_count':len(points),
                  'current_point_count':len(self.scans[-1][2]),
                  'sensor':'PhysX ray-cast LiDAR','intensity_model':'Native 0..255 hit indicator; not calibrated reflectance',
                  'scan_model':'instantaneous full scans; ten earlier scans registered with simulator poses'}
        return points,metadata
