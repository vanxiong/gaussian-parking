"""Isolated, frozen-pose camera renderer for continuous driving simulation.

The driving process owns physics and LiDAR. This process owns all six camera
render products and their blocking GPU readbacks. Requests contain the exact
car and dynamic-obstacle transforms belonging to the LiDAR snapshot.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    args=parser.parse_args()
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'multi_gpu':False,'width':640,'height':480})
    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf,UsdGeom
    from PIL import Image
    try:
        context=omni.usd.get_context()
        context.open_stage(str(args.scene.resolve()))
        while context.get_stage_loading_status()[2]>0:app.update()
        stage=context.get_stage()
        # Debug overlays never exist in camera inputs, independently of the
        # visibility of evidence dots in the driving viewport.
        stage.RemovePrim('/World/DrivingDebug')
        car=stage.GetPrimAtPath('/World/PhysXCar')
        transform=UsdGeom.Xformable(car)
        transform.ClearXformOpOrder()
        car_op=transform.AddTransformOp(opSuffix='sensorSnapshot')
        products=[]
        calibration=json.loads(args.calibration.read_text())
        for camera in calibration:
            rp=rep.create.render_product(camera['path'],tuple(camera['resolution']))
            rgb=rep.AnnotatorRegistry.get_annotator('rgb')
            rgb.attach([rp]);products.append((camera,rgb))
        rep.orchestrator.set_capture_on_play(False)
        rep.orchestrator.step(delta_time=0.,rt_subframes=3,pause_timeline=True)
        print('SENSOR_READY',flush=True)
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('shutdown'):break
            started=time.perf_counter()
            try:
                # USD uses row-vector matrices; the IPC calibration uses columns.
                car_op.Set(Gf.Matrix4d(np.asarray(request['world_from_ego']).T.tolist()))
                block=request.get('block')
                if block is None:
                    stage.RemovePrim('/World/TestParkingBlock')
                else:
                    cube=UsdGeom.Cube.Define(stage,'/World/TestParkingBlock')
                    cube.CreateSizeAttr(1.)
                    xf=UsdGeom.Xformable(cube)
                    xf.ClearXformOpOrder()
                    xf.AddTransformOp(opSuffix='sensorSnapshot').Set(Gf.Matrix4d(np.asarray(block).T.tolist()))
                    cube.CreateDisplayColorAttr([Gf.Vec3f(.9,.15,.1)])
                rep.orchestrator.step(delta_time=0.,rt_subframes=1,pause_timeline=True)
                cache=UsdGeom.XformCache()
                for camera,expected in zip(calibration,request['cameras']):
                    actual=np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(camera['path']))).T@np.diag([1,-1,-1,1])
                    target=np.asarray(request['world_from_ego'])@np.asarray(expected['ego_from_optical'])
                    if not np.allclose(actual,target,atol=1e-5,rtol=0):
                        raise RuntimeError('Sensor camera pose does not match LiDAR snapshot: '+camera['name'])
                directory=Path(request['capture'])
                for camera,rgb in products:
                    image=np.asarray(rgb.get_data())
                    if image.ndim!=3 or image.shape[2]<3 or image[:,:,:3].std()<2:
                        raise RuntimeError('Invalid sensor RGB: '+camera['name'])
                    Image.fromarray(image[:,:,:3].astype(np.uint8)).save(directory/(camera['name']+'.png'))
                response={'id':request['id'],'capture':str(directory),'seconds':time.perf_counter()-started}
            except Exception as error:
                response={'id':request['id'],'error':str(error)}
            print('SENSOR_RESULT '+json.dumps(response),flush=True)
    finally:
        app.close()


if __name__=='__main__':main()
