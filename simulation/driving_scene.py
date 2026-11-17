"""Closed-loop GaussianFormer3D/LiDAR parking: sense, replan, and control the PhysX sedan."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import queue
import subprocess
import threading
import time
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
PHYSICS_HZ=60
MAX_CRUISING_SPEED_MPS=6.
parser=argparse.ArgumentParser()
parser.add_argument('--headless',action='store_true')
parser.add_argument('--street',action='store_true',help='Start on the street and navigate to the parking destination')
parser.add_argument('--speed-cap',type=float,default=1.,help='Cruising cap in m/s (0.2 to 6; default 1); tight maneuvers stay slower')
parser.add_argument('--capture-only',action='store_true')
parser.add_argument('--evaluation-snapshots',action='store_true',help='Save exact per-capture USD ground truth independently of model inputs')
parser.add_argument('--blocked-goal',action='store_true')
parser.add_argument('--start-offset',type=float,default=0.)
parser.add_argument('--start-y-offset',type=float,default=0.)
parser.add_argument('--start-yaw-deg',type=float,default=None)
parser.add_argument('--goal-heading-deg',type=float,default=90.)
parser.add_argument('--heading-change-test',action='store_true',help='Automated GUI slider-drag regression')
parser.add_argument('--pose-controls-test',action='store_true',help='Exercise the GUI placement operation headlessly')
parser.add_argument('--max-sim-seconds',type=float,default=180.)
parser.add_argument('--screenshot',action='store_true')
parser.add_argument('--synchronous',action='store_true',help='Wait for each perception result (comparison mode)')
parser.add_argument('--max-perception-age',type=float,default=8.,help='Maximum capture age in wall seconds')
parser.add_argument('--worker-delay',type=float,default=0.,help='Test-only delay after the first model update')
parser.add_argument('--planner-delay',type=float,default=0.,help='Test-only delay on the first CPU search')
parser.add_argument('--block-at',type=float,default=-1.,help='Test-only dynamic bay block at simulation time')
parser.add_argument('--camera-width',type=int,default=640,choices=[480,640,960],help='Camera quality/performance setting; projections scale with resolution')
from async_parking import align_path, perception_stale
from system_status import driving_status
from destination_evidence import destination_state
args,_=parser.parse_known_args()
if args.start_yaw_deg is None:args.start_yaw_deg=90. if args.street else 0.
OUT=ROOT/'data/driving_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
OUT.mkdir(parents=True)
from isaacsim import SimulationApp
app=SimulationApp({'headless':args.headless,'width':1600,'height':1000,'multi_gpu':False})
from isaacsim.core.utils.extensions import enable_extension
enable_extension('omni.physx.vehicle');enable_extension('isaacsim.sensors.physx');app.update()
from pxr import Gf,Usd,UsdGeom,UsdPhysics,PhysxSchema,Vt
from isaacsim.core.api import World
from omni.kit.viewport.utility import get_active_viewport
from PIL import Image
import omni.replicator.core as rep
if args.street:
    from street_scene import build,overlaps,START,TARGET,INITIAL_YAW
else:
    from physx_sedan_scene import build,overlaps,START,TARGET
    INITIAL_YAW=0.

stage,car,colliders,calibration=build(OUT)
placement_settings=[args.start_offset,args.start_y_offset,args.start_yaw_deg]
final_heading=math.radians(args.goal_heading_deg)
desired_heading=final_heading
placement_drive=True;pose_control_events=[];placement_models=[];placement_error=''
body=stage.GetPrimAtPath('/World/PhysXCar/Appearance/Body')
bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_]).ComputeLocalBound(body).ComputeAlignedRange()
size=bounds.GetSize();footprint=[float(size[0]),float(size[1])]
if not 3<footprint[0]<6 or not 1<footprint[1]<3:raise RuntimeError('Unexpected authored body footprint: '+str(footprint))
controller=PhysxSchema.PhysxVehicleControllerAPI(car)
steering=PhysxSchema.PhysxVehicleSteeringAPI(car);steering.CreateAngleMultipliersAttr([1.,1.])
PhysxSchema.PhysxVehicleEngineAPI(car).GetPeakTorqueAttr().Set(300.)
world=World(physics_dt=1/PHYSICS_HZ,rendering_dt=1/PHYSICS_HZ,stage_units_in_meters=1,physics_prim_path='/World/Physics')
viewport=get_active_viewport();viewport.set_active_camera('/World/Overview')
from lidar_history import LidarHistory
lidar_history=LidarHistory(world,stage,car)

last_command={'direction':1,'throttle':0.,'brake':1.,'steer':0.}
def inputs(throttle=0.,brake=1.,delta=0.,direction=None):
    if direction is None:direction=last_command['direction']
    last_command.update(direction=int(direction),throttle=float(throttle),brake=float(brake),steer=float(delta))
    controller.GetAcceleratorAttr().Set(float(throttle));controller.GetBrake0Attr().Set(float(brake))
    controller.GetTargetGearAttr().Set(-1 if direction<0 else 1)
    controller.GetSteerAttr().Set(float(delta/.85))
    if abs(delta)<1e-5:mult=[1.,1.]
    else:
        radius=2.7/math.tan(delta)
        mult=[math.atan(2.7/(radius-.94))/delta,math.atan(2.7/(radius+.94))/delta]
    steering.GetAngleMultipliersAttr().Set(mult)

def pose():
    matrix=UsdGeom.XformCache().GetLocalToWorldTransform(car)
    p=matrix.ExtractTranslation();forward=matrix.TransformDir(Gf.Vec3d(1,0,0))
    heading=math.atan2(forward[1],forward[0])
    rear=np.array([p[0]-1.35*math.cos(heading),p[1]-1.35*math.sin(heading),heading])
    return p,heading,rear

def add_block():
    global needs_sensor_step
    needs_sensor_step=True
    block=UsdGeom.Cube.Define(stage,'/World/TestParkingBlock')
    block.GetSizeAttr().Set(1.)
    if not block.GetOrderedXformOps():block.AddTranslateOp();block.AddScaleOp()
    block.GetPrim().GetAttribute('xformOp:translate').Set(Gf.Vec3d(TARGET[0],TARGET[1],.6))
    block.GetPrim().GetAttribute('xformOp:scale').Set(Gf.Vec3f(2.,2.4,1.2))
    block.GetDisplayColorAttr().Set([Gf.Vec3f(.9,.15,.1)])
    UsdPhysics.CollisionAPI.Apply(block.GetPrim())
    colliders[:]=[item for item in colliders if item[0]!='/World/TestParkingBlock']
    center=np.array(TARGET[:2]);colliders.append(('/World/TestParkingBlock',center-[1.,1.2],center+[1.,1.2]))

def remove_block():
    stage.RemovePrim('/World/TestParkingBlock')
    colliders[:]=[item for item in colliders if item[0]!='/World/TestParkingBlock']
    resume()

world.reset();lidar_history.enable();inputs()
car.GetAttribute('xformOp:translate').Set(Gf.Vec3d(START[0]+args.start_offset,START[1]+args.start_y_offset,START[2]))
initial_yaw=math.radians(args.start_yaw_deg)
car.GetAttribute('xformOp:orient').Set(Gf.Quatf(math.cos(initial_yaw/2),0,0,math.sin(initial_yaw/2)))
for _ in range(240):
    world.step(render=True);lidar_history.sample()
body_clearance_height=float(UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_]).ComputeWorldBound(body).ComputeAlignedRange().GetMin()[2])-.01
if args.blocked_goal:add_block()
debug=UsdGeom.Xform.Define(stage,'/World/DrivingDebug')
products=[]
for camera in calibration['cameras']:
    camera=dict(camera)
    scale=args.camera_width/camera['resolution'][0]
    camera['resolution']=[args.camera_width,round(camera['resolution'][1]*scale)]
    intrinsic=np.asarray(camera['K']).copy();intrinsic[:2]*=scale;camera['K']=intrinsic.tolist()
    rp=rep.create.render_product(camera['path'],tuple(camera['resolution']))
    rgb=rep.AnnotatorRegistry.get_annotator('rgb')
    rgb.attach([rp]);products.append((camera,rp,rgb,None))
    rp.hydra_texture.set_updates_enabled(False)

def linux_path(path):
    value=Path(path).resolve().as_posix()
    return '/mnt/'+value[0].lower()+value[2:]

worker=subprocess.Popen(['wsl.exe','--distribution','Ubuntu-24.04','--exec','bash',linux_path(PROJECT/'tools/driving_worker.sh')],cwd=PROJECT,
    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,encoding='utf-8',errors='replace',
    creationflags=subprocess.CREATE_NO_WINDOW)
mailbox=queue.Queue()
control_commands=queue.Queue()
queued_heading=None;heading_changed_at=0.;heading_editing=False;heading_events=[]
def reader():
    with (OUT/'model_worker.log').open('w',encoding='utf-8') as log:
        for line in worker.stdout:log.write(line);log.flush();mailbox.put(line)
planner_process=subprocess.Popen([str(PROJECT/'.venv/Scripts/python.exe'),str(ROOT/'planning_worker.py')],cwd=ROOT,
    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,encoding='utf-8',errors='replace',
    creationflags=subprocess.CREATE_NO_WINDOW)
planner_mailbox=queue.Queue()
capture_context=None
def planner_reader():
    with (OUT/'planning_worker.log').open('w',encoding='utf-8') as log:
        for line in planner_process.stdout:log.write(line);log.flush();planner_mailbox.put(line)
threading.Thread(target=planner_reader,daemon=True).start()
planning_pending=None;planning_requested=True;plan_id=0;planning_updates=[];latest_map_directory=None;last_failed_plan_directory=None
threading.Thread(target=reader,daemon=True).start()
status='Loading GaussianFormer3D';label=None;system_label=None;system_detail=None;system_telemetry=None;planning_diagnostic=''
mode='auto';request_replan=True;restart=False;force_reset=True
cap=max(.2,min(MAX_CRUISING_SPEED_MPS,args.speed_cap));frame=0;elapsed=0.;last_capture=-100.;trajectory=[];perceptions=[]
pending=None;generation=0;map_sim_time=None;map_wall_time=None;needs_sensor_step=True
last_sensor_digest=None;last_sensor_pose=None
async_stats={'physics_steps_while_pending':0,'moving_steps_while_pending':0,'stale_brake_steps':0,'discarded_paths':0,'capture_seconds':[],'result_latencies':[],'frame_seconds':[],'readback_seconds':[]}
path=None;costmap=None;previous=np.array(pose()[0]);previous_delta=0.;speed_integral=0.;previous_desired=0.
def resume():
    global needs_sensor_step
    needs_sensor_step=True
    global request_replan,mode,path,generation,costmap,map_sim_time,map_wall_time,planning_requested,capture_context,last_failed_plan_directory,planning_diagnostic
    planning_diagnostic=''
    if stage.GetPrimAtPath('/World/DrivingDebug/RejectedClearance'):stage.RemovePrim('/World/DrivingDebug/RejectedClearance')
    capture_context=None;last_failed_plan_directory=None
    generation+=1;costmap=None;map_sim_time=map_wall_time=None
    request_replan=True;planning_requested=True;mode='auto';path=None;inputs()
    if not world.is_playing():world.play()
def stop():
    global mode
    mode='stopped';inputs()
def place_request(drive=False,original=False):
    global restart,placement_drive,desired_heading,capture_context
    capture_context=None
    desired_heading=final_heading
    update_goal_marker()
    stop();placement_drive=drive
    if original:
        placement_settings[:]=[0.,0.,math.degrees(INITIAL_YAW)]
        for model,value in zip(placement_models,placement_settings):model.set_value(value)
    restart=True

def enqueue_control(function,*arguments):
    # UI callbacks only record intent; never mutate USD or the timeline in a render update.
    control_commands.put((function,arguments))

def heading_changed(model):
    global queued_heading,heading_changed_at
    queued_heading=float(model.as_float);heading_changed_at=time.perf_counter()

def heading_begin_edit(model):
    global heading_editing
    heading_editing=True

def heading_end_edit(model):
    global heading_editing,heading_changed_at
    heading_changed(model);heading_editing=False;heading_changed_at=0.

def apply_queued_heading():
    global queued_heading,desired_heading,final_heading
    if queued_heading is None:return
    value=queued_heading;queued_heading=None
    if abs((math.radians(value)-final_heading+math.pi)%(2*math.pi)-math.pi)<1e-6:return
    final_heading=math.radians(value)
    desired_heading=final_heading
    update_goal_marker();resume()
    event={'heading_deg':value,'simulation_seconds':elapsed,'generation':generation}
    heading_events.append(event);print('HEADING_APPLIED',json.dumps(event),flush=True)

def update_goal_marker():
    curve=UsdGeom.BasisCurves.Define(stage,'/World/DrivingDebug/GoalHeading')
    curve.CreateTypeAttr('linear');curve.CreateCurveVertexCountsAttr([2])
    curve.CreatePointsAttr([Gf.Vec3f(TARGET[0],TARGET[1],.22),
        Gf.Vec3f(TARGET[0]+2*math.cos(desired_heading),TARGET[1]+2*math.sin(desired_heading),.22)])
    curve.CreateWidthsAttr([.09]);curve.SetWidthsInterpolation('constant')
    curve.CreateDisplayColorAttr([Gf.Vec3f(.3,1.,.3)])
def toggle_debug(name):
    prim=stage.GetPrimAtPath('/World/DrivingDebug/'+name)
    if prim:
        image=UsdGeom.Imageable(prim)
        if image.GetVisibilityAttr().Get()=='invisible':image.MakeVisible()
        else:image.MakeInvisible()
update_goal_marker()
if not args.headless:
    import omni.ui as ui
    status_panel=ui.Window('Driving system status',width=520,height=230)
    with status_panel.frame:
        with ui.VStack(spacing=8):
            system_label=ui.Label('INITIALIZING',height=30,style={'font_size':22})
            system_detail=ui.Label('Loading perception and preparing the simulation.',word_wrap=True,height=100)
            system_telemetry=ui.Label('',word_wrap=True,height=40)
    panel=ui.Window('GaussianFormer3D parking | car settings',width=520,height=850)
    with panel.frame:
        with ui.ScrollingFrame():
            with ui.VStack(spacing=8):
                label=ui.Label(status,word_wrap=True,height=60)
                ui.Label('GaussianFormer3D semantic cost + LiDAR collision clearance',word_wrap=True,height=32)
                ui.Label('Background inference/planning; brakes if sensor evidence gets stale.',word_wrap=True,height=32)
                ui.Label('Cruising speed cap (m/s); turns / parking slow down',word_wrap=True,height=32)
                speed_model=ui.FloatSlider(min=.2,max=MAX_CRUISING_SPEED_MPS,height=24).model;speed_model.set_value(cap)
                def change_speed(model):
                    global cap
                    cap=max(.2,min(MAX_CRUISING_SPEED_MPS,model.as_float))
                speed_model.add_value_changed_fn(change_speed)
                ui.Separator(height=8)
                ui.Label('Starting pose offsets | horizontal plane is X / Y (Z is up)',word_wrap=True,height=34)
                for index,title,low,high in [(0,'X offset from original start (m)',-8.,8.),
                    (1,'Y offset from original start (m)',-8.,8.),
                    (2,'Car heading (degrees): 0 east, 90 north',-180.,180.)]:
                    ui.Label(title,height=22)
                    model=ui.FloatSlider(min=low,max=high,height=24).model
                    model.set_value(placement_settings[index]);placement_models.append(model)
                    model.add_value_changed_fn(lambda value,index=index:placement_settings.__setitem__(index,value.as_float))
                ui.Label('Adjust the sliders, then place the car. Overlapping placements are rejected.',word_wrap=True,height=34)
                ui.Button('Place car here (stay stopped)',clicked_fn=lambda:enqueue_control(place_request,False),height=26)
                ui.Button('Place car here and park',clicked_fn=lambda:enqueue_control(place_request,True),height=26)
                ui.Button('Reset to original pose and park',clicked_fn=lambda:enqueue_control(place_request,True,True),height=26)
                ui.Separator(height=8)
                ui.Label('Desired parking heading (degrees): 0 = +X, 90 = +Y',word_wrap=True,height=34)
                heading_model=ui.FloatSlider(min=-180,max=180,height=24).model
                heading_model.set_value(math.degrees(final_heading));heading_model.add_value_changed_fn(heading_changed)
                heading_model.add_begin_edit_fn(heading_begin_edit);heading_model.add_end_edit_fn(heading_end_edit)
                ui.Label('One final destination. Green line marks the desired parking heading.',word_wrap=True,height=34)
                ui.Button('Stop',clicked_fn=lambda:enqueue_control(stop),height=26)
                ui.Button('Replan from current position / resume',clicked_fn=lambda:enqueue_control(resume),height=26)
                ui.Button('Block parking bay and replan',clicked_fn=lambda:(enqueue_control(add_block),enqueue_control(resume)),height=26)
                ui.Button('Remove test block and replan',clicked_fn=lambda:enqueue_control(remove_block),height=26)
                ui.Button('Overview',clicked_fn=lambda:enqueue_control(viewport.set_active_camera,'/World/Overview'),height=26)
                ui.Button('Toggle GaussianFormer3D evidence (purple)',clicked_fn=lambda:enqueue_control(toggle_debug,'GaussianEvidence'),height=26)
                ui.Button('Toggle LiDAR obstacles (yellow)',clicked_fn=lambda:enqueue_control(toggle_debug,'LidarEvidence'),height=26)
                ui.Button('Toggle rejected clearance (red / amber)',clicked_fn=lambda:enqueue_control(toggle_debug,'RejectedClearance'),height=26)

slider_test_finished=False
if args.heading_change_test:
    if args.headless:raise ValueError('Heading regression uses the GUI; omit --headless')
    import omni.kit.app
    slider_test_count=0;slider_test_started=None;slider_test_last=0.
    def slider_regression_tick(event):
        global slider_test_count,slider_test_started,slider_test_last,slider_test_finished
        if elapsed<3. or slider_test_finished:return
        now=time.perf_counter()
        if now-slider_test_last<.025:return
        slider_test_last=now
        if slider_test_started is None:
            slider_test_started=now;heading_model.begin_edit()
        value=85.+25.*math.sin(slider_test_count*.7)
        heading_model.set_value(value);slider_test_count+=1
        if slider_test_count>=40:
            heading_model.set_value(85.);heading_model.end_edit();slider_test_finished=True
            print('HEADING_SLIDER_TEST_FINISHED',slider_test_count,flush=True)
    slider_test_subscription=omni.kit.app.get_app().get_post_update_event_stream().create_subscription_to_pop(slider_regression_tick,name='Heading slider regression')

def wait_for(prefix,timeout=180.):
    started=time.perf_counter()
    while app.is_running() and time.perf_counter()-started<timeout:
        try:
            line=mailbox.get_nowait()
            if line.startswith(prefix):return line[len(prefix):].strip()
        except queue.Empty:pass
        if worker.poll() is not None:raise RuntimeError('Model worker exited; see '+str(OUT/'model_worker.log'))
        if label:label.text=status
        app.update();time.sleep(.002)
    raise RuntimeError('Model worker timeout')

def submit_capture():
    global frame,status,force_reset,last_capture,pending,last_sensor_digest,last_sensor_pose
    capture_started=time.perf_counter()
    frame+=1;directory=OUT/f'frame_{frame:03d}';directory.mkdir()
    if args.synchronous:world.pause()
    UsdGeom.Imageable(debug).MakeInvisible()
    cache=UsdGeom.XformCache();ego=np.array(cache.GetLocalToWorldTransform(car)).T
    lidar_points,lidar_metadata=lidar_history.snapshot(ego)
    cameras=[]
    for record,rp,rgb,_ in products:
        rp.hydra_texture.set_updates_enabled(True)
        optical=np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(record['path']))).T@np.diag([1,-1,-1,1])
        local=np.linalg.inv(ego)@optical;k=np.asarray(record['K'])
        cameras.append({'name':record['name'],'image':record['name']+'.png','resolution':record['resolution'],
                        'intrinsic':k.tolist(),'ego_from_optical':local.tolist(),
                        'projection_from_ego':(k@np.linalg.inv(local)[:3]).tolist()})
    rep.orchestrator.step(delta_time=0.,rt_subframes=1,pause_timeline=args.synchronous)
    snapshots=[];digest=hashlib.sha256()
    for record,rp,rgb,_ in products:
        pixels=np.asarray(rgb.get_data())
        if pixels.ndim!=3 or pixels.shape[2]<3:raise RuntimeError('Invalid RGB capture')
        image=pixels[:,:,:3].astype(np.uint8)
        if image.std()<2:raise RuntimeError('Blank RGB capture')
        snapshots.append((record['name'],image));digest.update(image.tobytes())
        rp.hydra_texture.set_updates_enabled(False)
    after=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(car)).T
    if not np.allclose(ego,after,atol=1e-6,rtol=0):raise RuntimeError('Vehicle moved during synchronized RGB/LiDAR snapshot')
    current=pose()[2];sensor_digest=digest.hexdigest()
    if last_sensor_digest==sensor_digest and last_sensor_pose is not None and np.linalg.norm(current[:2]-last_sensor_pose[:2])>.15:
        raise RuntimeError('RGB annotations did not refresh while the car moved')
    last_sensor_digest=sensor_digest;last_sensor_pose=current.copy()
    metadata={'world_from_ego':ego.tolist(),'cameras':cameras,'lidar':lidar_metadata,
              'simulation_seconds':elapsed,'simulation_timestamp_seconds':float(world.current_time),
              'capture_mode':'synchronized zero-delta RGB/LiDAR snapshot; background model inference'}
    if args.evaluation_snapshots:
        gt_started=time.perf_counter()
        stage.GetRootLayer().Export(str(directory/'scene.usda'))
        metadata['ground_truth_snapshot']={'scene':'scene.usda','timestamp':float(world.current_time),
            'export_seconds':time.perf_counter()-gt_started,'used_as_model_input':False}
    request={'id':frame,'capture':str(directory),'start':current.tolist(),'footprint':footprint,
             'body_clearance_height':body_clearance_height,'reset':force_reset,
             'test_delay':args.worker_delay if frame==2 else 0.}
    force_reset=False;status=f'BACKGROUND GAUSSIANFORMER3D | frame {frame}'
    pending={'id':frame,'directory':directory,'generation':generation,'sim_time':elapsed,'wall_time':capture_started}
    last_capture=elapsed
    def save_and_send():
        try:
            for name,image in snapshots:Image.fromarray(image).save(directory/(name+'.png'))
            np.save(directory/'lidar.npy',lidar_points)
            (directory/'capture.json').write_text(json.dumps(metadata,indent=2))
            worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush()
        except Exception as error:
            mailbox.put('WORKER_RESULT '+json.dumps({'id':request['id'],'error':'Capture writer: '+str(error)}))
    if args.synchronous:save_and_send()
    else:threading.Thread(target=save_and_send,daemon=True).start()
    UsdGeom.Imageable(debug).MakeVisible()
    async_stats['capture_seconds'].append(time.perf_counter()-capture_started)
    if args.synchronous:
        accept_response(json.loads(wait_for('WORKER_RESULT ')))
        world.play()


def accept_response(response):
    global path,costmap,pending,map_sim_time,map_wall_time,mode,request_replan,latest_map_directory,planning_requested
    job=pending;pending=None
    if job is None or response.get('id')!=job['id']:raise RuntimeError('Unexpected worker result ID')
    if job['generation']!=generation:return
    if response.get('error'):raise RuntimeError(response['error'])
    directory=job['directory'];perceptions.append(response)
    latency=time.perf_counter()-job['wall_time'];response['latency_wall_seconds']=latency
    async_stats['result_latencies'].append(latency)
    costmap=dict(np.load(directory/'costmap.npz',allow_pickle=False))
    map_sim_time=job['sim_time'];map_wall_time=job['wall_time']
    latest_map_directory=directory
    if path is not None:
        # Refresh clearance independently of CPU search; retain a still-valid route.
        gear=int(path[0,3]);changes=np.flatnonzero(path[:,3]!=gear)
        end=int(changes[0]) if len(changes) else len(path)
        near=int(np.argmin(np.linalg.norm(path[:end,:2]-pose()[2][:2],axis=1)))
        if not known_route_clear(path[near:]):
            planning_requested=True;path=None;inputs()
    if perception_stale(elapsed,time.perf_counter(),map_sim_time,map_wall_time,args.max_perception_age):request_replan=True
    labels=np.load(directory/'labels.npy',allow_pickle=False);confidence=np.load(directory/'confidence.npy',allow_pickle=False)
    indices=np.argwhere((labels>0)&~np.isin(labels,[11,12,13,14])&(confidence>=.35))
    xyz=(indices+.5)*[.5,.5,.5]+[-50.,-50.,-5.]
    xyz=np.c_[xyz,np.ones(len(xyz))]@np.asarray(response['world_from_reference']).T
    xyz=xyz[(xyz[:,2]>.3)&(xyz[:,2]<2.5)]
    if len(xyz)>3000:xyz=xyz[::math.ceil(len(xyz)/3000)]
    for name,points,color,width in [('GaussianEvidence',xyz[:,:3],(.7,.2,1.),.16)]:
        dots=UsdGeom.Points.Define(stage,'/World/DrivingDebug/'+name)
        dots.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        dots.GetWidthsAttr().Set([width]);dots.SetWidthsInterpolation('constant')
        dots.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    cells=np.argwhere(costmap['obstacle'])
    xy=cells[:,::-1]*float(costmap['res'])+costmap['origin']+float(costmap['res'])/2
    dots=UsdGeom.Points.Define(stage,'/World/DrivingDebug/LidarEvidence')
    dots.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.c_[xy,np.full(len(xy),.12)].astype(np.float32)))
    dots.GetWidthsAttr().Set([.1]);dots.SetWidthsInterpolation('constant')
    dots.GetDisplayColorAttr().Set([Gf.Vec3f(1.,.7,.1)])
    draw_path()
    UsdGeom.Imageable(debug).MakeVisible()
    print('PERCEPTION_UPDATE',job['id'],json.dumps(response.get('metrics')),'age',round(latency,3),flush=True)


def draw_path():
    if path is not None and len(path)>1:
        curve=UsdGeom.BasisCurves.Define(stage,'/World/DrivingDebug/PlannedPath')
        curve.CreateTypeAttr('linear');curve.CreateCurveVertexCountsAttr([len(path)])
        curve.CreatePointsAttr([Gf.Vec3f(float(x),float(y),.16) for x,y,_,_ in path])
        curve.CreateWidthsAttr([.06]);curve.SetWidthsInterpolation('constant')
        curve.CreateDisplayColorAttr([Gf.Vec3f(.1,.9,1.)])
    elif stage.GetPrimAtPath('/World/DrivingDebug/PlannedPath'):stage.RemovePrim('/World/DrivingDebug/PlannedPath')


def submit_planning():
    global planning_pending,planning_requested,plan_id
    if latest_map_directory is None:return
    plan_id+=1
    goal=[TARGET[0]-1.35*math.cos(desired_heading),TARGET[1]-1.35*math.sin(desired_heading),desired_heading]
    request={'id':plan_id,'capture':str(latest_map_directory),'start':pose()[2].tolist(),'goal':goal,'footprint':footprint,
             'test_delay':args.planner_delay if plan_id==1 else 0.,'timeout':40.,'road_layout':str(ROOT/'road_layout.json')}
    if not args.street:request.pop('road_layout',None)
    planning_pending={'id':plan_id,'generation':generation,'wall_time':time.perf_counter(),'request':request}
    planning_requested=False
    planner_process.stdin.write(json.dumps(request)+'\n');planner_process.stdin.flush()


def accept_plan(response):
    global path,planning_pending,planning_requested,mode,last_failed_plan_directory,planning_diagnostic
    job=planning_pending;planning_pending=None
    if job is None or response.get('id')!=job['id']:raise RuntimeError('Unexpected planner result ID')
    if job['generation']!=generation:return
    if response.get('error'):raise RuntimeError(response['error'])
    response['latency_wall_seconds']=time.perf_counter()-job['wall_time'];planning_updates.append(response)
    candidate=None if response['path'] is None else np.asarray(response['path'])
    detail=response['planning'].get('clearance_diagnostics',{}) if candidate is None else {}
    planning_diagnostic=''
    rejected=UsdGeom.Xform.Define(stage,'/World/DrivingDebug/RejectedClearance')
    for name,key,color in [('Blocked','blocked_xy',(.95,.1,.1)),('Unobserved','unobserved_xy',(1.,.65,.1))]:
        dots=UsdGeom.Points.Define(stage,str(rejected.GetPath())+'/'+name)
        xy=np.asarray(detail.get(key,[]),dtype=np.float32).reshape(-1,2)
        dots.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.c_[xy,np.full(len(xy),.25)].astype(np.float32)))
        dots.CreateWidthsAttr([.16]*len(xy));dots.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if candidate is None:
        planning_diagnostic=response['planning']['reason']
        if response['planning'].get('termination'):planning_diagnostic+=' ('+response['planning']['termination']+')'
        if detail:
            planning_diagnostic+=f' | footprint: {detail["obstacle_cells"]} obstacle cells, {detail["unobserved_cells"]} unobserved, {detail["out_of_bounds_cells"]} outside map'
    if candidate is None:
        last_failed_plan_directory=job['request']['capture']
        # The older snapshot may differ from the current map; retry before stopping.
        if path is None and response['planning'].get('goal_rejection')=='obstacle' and not known_obstacle_clear(job['request']['goal']):
            mode='no_safe_path';inputs()
        else:planning_requested=True
    else:
        aligned=align_path(candidate,pose()[2])
        # A delayed route is accepted only against current, independently refreshed evidence.
        if aligned is not None and known_route_clear(aligned):
            path=aligned
        else:
            async_stats['discarded_paths']+=1;planning_requested=True
    draw_path()
    print('PLANNER_UPDATE',job['id'],json.dumps(response['planning']),'accepted',path is not None,flush=True)


def measured_clear(rear,require_observed=True):
    if costmap is None:return False
    a=np.arange(-footprint[0]/2-.12,footprint[0]/2+.13,.15)+1.35
    b=np.arange(-footprint[1]/2-.12,footprint[1]/2+.13,.15)
    xx,yy=np.meshgrid(a,b);yaw=rear[2]
    xy=np.stack([rear[0]+xx*math.cos(yaw)-yy*math.sin(yaw),rear[1]+xx*math.sin(yaw)+yy*math.cos(yaw)],-1).reshape(-1,2)
    ij=np.floor((xy-costmap['origin'])/float(costmap['res'])).astype(int)
    if (ij<0).any() or (ij[:,0]>=costmap['free'].shape[1]).any() or (ij[:,1]>=costmap['free'].shape[0]).any():return False
    return not costmap['obstacle'][ij[:,1],ij[:,0]].any() and (not require_observed or costmap['free'][ij[:,1],ij[:,0]].mean()>=.92)

def known_obstacle_clear(rear):
    return measured_clear(rear,require_observed=False)

def known_route_clear(points):
    if costmap is None:return False
    points=np.asarray(points)
    a=np.arange(-footprint[0]/2-.12,footprint[0]/2+.13,.15)+1.35
    b=np.arange(-footprint[1]/2-.12,footprint[1]/2+.13,.15)
    xx,yy=np.meshgrid(a,b);xx=xx.ravel()[None,:];yy=yy.ravel()[None,:]
    c=np.cos(points[:,2,None]);s=np.sin(points[:,2,None])
    x=points[:,0,None]+xx*c-yy*s;y=points[:,1,None]+xx*s+yy*c
    ix=np.floor((x-costmap['origin'][0])/float(costmap['res'])).astype(int)
    iy=np.floor((y-costmap['origin'][1])/float(costmap['res'])).astype(int)
    if (ix<0).any() or (iy<0).any() or (ix>=costmap['free'].shape[1]).any() or (iy>=costmap['free'].shape[0]).any():return False
    return not costmap['obstacle'][iy,ix].any()


try:
    world.pause();wait_for('WORKER_READY')
    # Initialize the six annotator graphs before reading them from normal renders.
    for _,rp,_,_ in products:rp.hydra_texture.set_updates_enabled(True)
    rep.orchestrator.step(delta_time=0.,rt_subframes=3,pause_timeline=True)
    for _,rp,_,_ in products:rp.hydra_texture.set_updates_enabled(False)
    rep.orchestrator.set_capture_on_play(False)
    world.play()
    block_injected=False;run_started=time.perf_counter();step_index=0
    if args.pose_controls_test:place_request(False)
    while app.is_running():
        step_started=time.perf_counter()
        while not control_commands.empty():
            function,arguments=control_commands.get_nowait();function(*arguments)
        if queued_heading is not None and not heading_editing and time.perf_counter()-heading_changed_at>=.3:
            apply_queued_heading()
        if restart:
            requested_position=[START[0]+placement_settings[0],START[1]+placement_settings[1],START[2]]
            requested_yaw=math.radians(placement_settings[2])
            obstacle=overlaps(requested_position,requested_yaw,colliders)
            restart=False
            if obstacle:
                placement_error=obstacle
                mode='placement_blocked';inputs()
                status='PLACEMENT BLOCKED: '+obstacle
                pose_control_events.append({'accepted':False,'position':requested_position,'yaw_deg':placement_settings[2],'obstacle':obstacle})
            else:
                placement_error=''
                generation+=1;path=None;costmap=None;map_sim_time=map_wall_time=None;planning_requested=True;latest_map_directory=None
                world.stop();inputs();lidar_history.reset();world.reset();lidar_history.enable();needs_sensor_step=True
                car.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*requested_position))
                car.GetAttribute('xformOp:orient').Set(Gf.Quatf(math.cos(requested_yaw/2),0,0,math.sin(requested_yaw/2)))
                UsdPhysics.RigidBodyAPI(car).GetVelocityAttr().Set(Gf.Vec3f(0));UsdPhysics.RigidBodyAPI(car).GetAngularVelocityAttr().Set(Gf.Vec3f(0))
                for _ in range(PHYSICS_HZ):
                    world.step(render=True);lidar_history.sample()
                force_reset=True;request_replan=placement_drive;mode='auto' if placement_drive else 'stopped'
                previous=np.array(pose()[0]);previous_delta=previous_desired=speed_integral=0.
                last_sensor_digest=last_sensor_pose=None
                actual,actual_yaw,_=pose()
                event={'accepted':True,'position':list(actual),'yaw_deg':math.degrees(actual_yaw),'requested_position':requested_position,
                    'requested_yaw_deg':placement_settings[2],'drive':placement_drive,'goal_heading_deg':math.degrees(desired_heading)}
                pose_control_events.append(event);print('POSE_PLACED',json.dumps(event),flush=True)
                if args.pose_controls_test:
                    distance=math.hypot(actual[0]-TARGET[0],actual[1]-TARGET[1]);angle=abs((actual_yaw-desired_heading+math.pi)%(2*math.pi)-math.pi)
                    speed=0.;collision=overlaps(actual,actual_yaw,colliders);mode='stopped';break
        while not mailbox.empty():
            line=mailbox.get_nowait()
            if line.startswith('WORKER_RESULT '):accept_response(json.loads(line[len('WORKER_RESULT '):]))
        while not planner_mailbox.empty():
            planner_line=planner_mailbox.get_nowait()
            if planner_line.startswith('PLANNER_RESULT '):accept_plan(json.loads(planner_line[len('PLANNER_RESULT '):]))
        if planner_process.poll() is not None:raise RuntimeError('Path-search worker exited')
        if planning_pending and time.perf_counter()-planning_pending['wall_time']>65:raise RuntimeError('Path-search worker timeout')
        if worker.poll() is not None:raise RuntimeError('Perception worker exited')
        if pending and time.perf_counter()-pending['wall_time']>60:raise RuntimeError('Perception worker timeout')
        if args.block_at>=0 and not block_injected and elapsed>=args.block_at:
            add_block();block_injected=True
        if mode in ('auto','no_safe_path') and not needs_sensor_step and pending is None and (request_replan or elapsed-last_capture>=1.):
            if request_replan:planning_requested=True
            request_replan=False
            submit_capture()
        if mode=='auto' and planning_pending is None and costmap is not None and str(latest_map_directory)!=last_failed_plan_directory and (planning_requested or path is None):
            submit_planning()
        if args.capture_only and perceptions:break
        p,heading,rear=pose();speed=float((np.array(p)-previous)[:2]@np.array([math.cos(heading),math.sin(heading)])*PHYSICS_HZ);previous=np.array(p)
        distance=math.hypot(p[0]-TARGET[0],p[1]-TARGET[1]);angle=abs((heading-desired_heading+math.pi)%(2*math.pi)-math.pi)
        stale=perception_stale(elapsed,time.perf_counter(),map_sim_time,map_wall_time,args.max_perception_age)
        goal_rear=[TARGET[0]-1.35*math.cos(desired_heading),TARGET[1]-1.35*math.sin(desired_heading),desired_heading]
        destination=destination_state(costmap,goal_rear,footprint)
        if mode=='no_safe_path' and not stale and destination in ('clear','unobserved'):
            mode='auto';path=None;planning_requested=True;request_replan=True
        if mode=='auto' and distance<.2 and angle<math.radians(7) and abs(speed)<.15:
            mode='parked';inputs()
        hold_reason='stale/missing sensor evidence' if mode=='auto' and stale else 'destination blocked by obstacle evidence' if mode=='no_safe_path' and destination=='blocked' else 'waiting for a path' if path is None else ''
        if mode=='auto' and path is not None and not stale:
            direction=int(path[0,3])
            changes=np.flatnonzero(path[:,3]!=direction)
            segment_end=int(changes[0]-1) if len(changes) else len(path)-1
            segment_distance=float(np.linalg.norm(path[segment_end,:2]-rear[:2]))
            near=int(np.argmin(np.linalg.norm(path[:segment_end+1,:2]-rear[:2],axis=1)))
            end_heading=path[segment_end,2]
            end_delta=path[segment_end,:2]-rear[:2]
            segment_remaining=float(end_delta@(direction*np.array([math.cos(end_heading),math.sin(end_heading)])))
            end_lateral=abs(float(end_delta@np.array([-math.sin(end_heading),math.cos(end_heading)])))
            cusp_reached=segment_distance<.25 or (near>=segment_end-5 and segment_remaining<=.15 and end_lateral<.75)
            # Stop at the cusp, including recovery when its endpoint was passed.
            if len(changes) and cusp_reached and abs(speed)<.12:
                path=path[segment_end+1:];direction=int(path[0,3])
                changes=np.flatnonzero(path[:,3]!=direction)
                segment_end=int(changes[0]-1) if len(changes) else len(path)-1
                segment_distance=float(np.linalg.norm(path[segment_end,:2]-rear[:2]))
                previous_desired=speed_integral=0.
            near=int(np.argmin(np.linalg.norm(path[:segment_end+1,:2]-rear[:2],axis=1)))
            look=near;lookahead=1.2+.35*abs(speed)
            while look<segment_end and np.linalg.norm(path[look,:2]-rear[:2])<lookahead:look+=1
            aim=path[look,:2].copy()
            if look==len(path)-1:aim+=direction*max(0.,lookahead-np.linalg.norm(aim-rear[:2]))*np.array([math.cos(desired_heading),math.sin(desired_heading)])
            d=aim-rear[:2];local_y=-math.sin(heading)*d[0]+math.cos(heading)*d[1]
            delta=float(np.clip(math.atan2(5.4*local_y,max(float(d@d),.1)),-.65,.65))
            delta=float(np.clip(delta,previous_delta-.55/PHYSICS_HZ,previous_delta+.55/PHYSICS_HZ));previous_delta=delta
            # Slow before upcoming route curvature, rather than at authored mission stops.
            preview=path[near:min(segment_end+1,near+max(15,int((abs(speed)**2/1.8+2.)/.1)))]
            curvature=0.
            if len(preview)>1:
                heading_steps=(np.diff(preview[:,2])+math.pi)%(2*math.pi)-math.pi
                lengths=np.linalg.norm(np.diff(preview[:,:2],axis=0),axis=1)
                curvature=float(np.max(abs(heading_steps)/np.maximum(lengths,.01)))
            maneuver_cap=min(cap,1.) if distance<10. or direction<0 or curvature>.08 else cap
            if destination=='unobserved':maneuver_cap=min(maneuver_cap,1.)
            desired=min(maneuver_cap,math.sqrt(.6/max(abs(math.tan(delta)/2.7),.001)),max(.04,distance*.6))
            if len(changes):
                desired=min(desired,max(.025,segment_distance*.6))
                if near>=segment_end-15:
                    desired=min(desired,math.sqrt(1.8*max(segment_remaining-.12,0.)))
            desired=float(np.clip(desired,previous_desired-1.8/PHYSICS_HZ,previous_desired+2./PHYSICS_HZ));previous_desired=desired
            cruising=maneuver_cap>1. and direction>0 and cap>1.
            error=desired-direction*speed
            speed_integral=float(np.clip(speed_integral+error/PHYSICS_HZ,0,4. if cruising else 1.5))
            throttle=min(.85 if cruising else .35,max(0.,error*.3+speed_integral*.12));brake=min(1.,max(0.,-error*.6))
            # Native drivetrain may stick during a low-speed reverse steering launch.
            # Stop assisting as soon as motion starts; speed/clearance guards follow.
            if direction<0 and abs(speed)<.05 and desired>.03 and brake==0.:
                throttle=max(throttle,.40)
            if error<-.08:throttle=0.
            if len(changes) and near>=segment_end-5 and segment_remaining<=.15:
                throttle=0.;brake=1.;hold_reason='stopping for gear change'
            if speed*direction<-.1:throttle=0.;brake=1.
            predicted=rear.copy();horizon=abs(speed)**2/(2*.9)+abs(speed)*.25+.05
            clearance=True
            for _ in range(max(1,math.ceil(horizon/.05))):
                ds=direction*horizon/max(1,math.ceil(horizon/.05))
                middle=predicted[2]+ds*math.tan(delta)/2.7/2
                predicted[:2]+=ds*np.array([math.cos(middle),math.sin(middle)])
                predicted[2]+=ds*math.tan(delta)/2.7
                if not measured_clear(predicted):clearance=False;break
            if not clearance:
                throttle=0.;brake=1.;request_replan=True;planning_requested=True
                hold_reason='obstacle in braking path' if not measured_clear(predicted,require_observed=False) else 'waiting to observe braking path'
            if distance<.2 and angle<math.radians(7) and abs(speed)<.15:mode='parked';throttle=0.;brake=1.
            inputs(throttle,brake,delta,direction)
        else:
            inputs()
            if stale and path is not None:
                async_stats['stale_brake_steps']+=1;request_replan=True
                previous_desired=speed_integral=0.
        if pending:
            async_stats['physics_steps_while_pending']+=1
            if abs(speed)>.1:async_stats['moving_steps_while_pending']+=1
        collision=overlaps(p,heading,colliders)
        if collision:mode='collision_audit_stop';inputs()
        planning_seconds=None if planning_pending is None else time.perf_counter()-planning_pending['wall_time']
        activity,activity_detail,activity_color=driving_status(mode,has_path=path is not None,stale=stale,
            planning_seconds=planning_seconds,hold_reason=hold_reason,destination=destination)
        if planning_diagnostic and (path is None or mode=='no_safe_path'):
            activity_detail+=' Last search: '+planning_diagnostic
        if step_index%(PHYSICS_HZ//4)==0:trajectory.append({'t':elapsed,'position':list(p),'heading':heading,'speed':speed,'mode':mode,
            'perception_stale':stale,'pending_id':None if pending is None else pending['id'],
            'command':dict(last_command),'hold_reason':hold_reason,'planning_pending':planning_pending is not None,'destination_state':destination})
        if step_index%PHYSICS_HZ==0:
            (OUT/'live_status.json').write_text(json.dumps({'t':elapsed,'position':list(p),'heading':heading,'speed':speed,
                'mode':mode,'command':dict(last_command),'hold_reason':hold_reason,'planning_pending':planning_pending is not None,
                'system_status':activity,'system_status_detail':activity_detail,'planning_elapsed_seconds':planning_seconds,'destination_state':destination}))
        age=0. if map_wall_time is None else time.perf_counter()-map_wall_time
        gear_text='R' if last_command['direction']<0 else 'D'
        status=f'NAVIGATE TO PARKING GOAL | {activity} | {gear_text} | {speed:+.2f} m/s | target {distance:.2f} m\nCamera age {age:.1f}s | throttle {last_command["throttle"]:.2f} | brake {last_command["brake"]:.2f}'
        if hold_reason:status+='\n'+hold_reason
        if mode=='placement_blocked':status+='\nChoose another pose: '+placement_error
        if label:label.text=status
        if system_label is not None:
            system_label.text=activity
            system_label.style={'font_size':22,'color':{'red':0xFF6666FF,'amber':0xFF66CCFF,'green':0xFF88DD88,'blue':0xFFFFCC88}[activity_color]}
            system_detail.text=activity_detail+(' '+placement_error if mode=='placement_blocked' else '')
            camera_age='not available' if map_wall_time is None else f'{age:.1f}s'
            system_telemetry.text=f'{gear_text} | Speed {speed:+.2f} m/s | Goal {distance:.2f} m\nCamera age: {camera_age}'
        if (args.headless or (args.heading_change_test and slider_test_finished and queued_heading is None)) and ((mode=='collision_audit_stop' or (mode=='parked' and abs(speed)<.03) or (mode=='no_safe_path' and abs(speed)<.05)) or elapsed>args.max_sim_seconds):break
        if capture_context is not None:UsdGeom.Imageable(debug).MakeInvisible()
        world.step(render=step_index%2==0);elapsed+=1/PHYSICS_HZ;step_index+=1
        lidar_history.sample();needs_sensor_step=False
        async_stats['frame_seconds'].append(time.perf_counter()-step_started)
        time.sleep(max(0.,1/PHYSICS_HZ-(time.perf_counter()-step_started)))
    report={'status':mode,'position_error_m':distance if not args.capture_only else None,'heading_error_deg':math.degrees(angle) if not args.capture_only else None,
        'final_destination':[TARGET[0],TARGET[1],desired_heading],'preset_intermediate_goals':0,'footprint':footprint,'simulation_seconds':elapsed,'trajectory':trajectory,'perception_updates':perceptions,'planning_updates':planning_updates,
        'scenario':'street_to_parking' if args.street else 'parking',
        'planner_inputs':('pavement layout and lane costs; ' if args.street else '')+'LiDAR obstacles, GaussianFormer3D semantic cost, one final goal, simulator localization; no collider map',
        'ground_truth_collision_watchdog':True,'collision_audit_obstacle':collision if not args.capture_only else None,
        'body_clearance_height_m':body_clearance_height,
        'pose_control_events':pose_control_events,'heading_events':heading_events,'desired_heading_deg':math.degrees(desired_heading),
        'final_speed_mps':None if args.capture_only else speed,
        'safety_mode':'LiDAR-assisted GaussianFormer3D driving; simulator localization',
        'async_mode':not args.synchronous,'capture_mode':'synchronized RGB/LiDAR snapshots','async_metrics':async_stats,
        'controller_hz':PHYSICS_HZ,'viewport_target_hz':PHYSICS_HZ//2,
        'camera_resolution':products[0][0]['resolution'],'cruising_speed_cap_mps':cap,
        'asynchronous_rendering':False,'camera_batch_size':6,
        'peak_speed_mps':max((abs(q['speed']) for q in trajectory),default=0.),
        'run_wall_seconds':time.perf_counter()-run_started,'heading_slider_test_finished':slider_test_finished}
    (OUT/'result.json').write_text(json.dumps(report,indent=2));stage.GetRootLayer().Export(str(OUT/'scene.usda'))
    if args.screenshot:
        world.pause()
        # The exported still uses a synchronous render after draining queued frames.
        import carb
        carb.settings.get_settings().set_bool('/app/asyncRendering',False)
        for _ in range(3):app.update()
        product=rep.create.render_product('/World/Overview',(1280,800))
        rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product])
        for _ in range(3):
            rep.orchestrator.step(delta_time=0.,rt_subframes=4,pause_timeline=True)
            data=np.asarray(rgb.get_data())
            if data.ndim==3 and data.shape[2]>=3:break
        if data.ndim!=3:raise RuntimeError('Final overview renderer did not return an image')
        pixels=data[:,:,:3].astype(np.uint8)
        if pixels.std()<2:raise RuntimeError('Invalid final overview')
        Image.fromarray(pixels).save(OUT/'overview.png')
    (ROOT/'data/latest_driving_run.json').write_text(json.dumps({'run':str(OUT)},indent=2))
    console_report={k:v for k,v in report.items() if k not in ['trajectory','perception_updates','planning_updates']}
    console_report['async_metrics']={k:v for k,v in async_stats.items() if not isinstance(v,list)}
    console_report['frame_ms_p95_p99_max']=np.percentile(np.asarray(async_stats['frame_seconds'][180:] or async_stats['frame_seconds'])*1000,[95,99,100]).tolist()
    print('GAUSSIANFORMER3D_DRIVING_RESULT',json.dumps(console_report),flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    raise
finally:
    inputs();world.stop()
    for process in (worker,planner_process):
        if process.poll() is None:
            try:
                process.stdin.close();process.wait(timeout=10)
            except (subprocess.TimeoutExpired,OSError):
                process.terminate();process.wait(timeout=10)
    app.close()
if args.headless and not args.capture_only and not args.pose_controls_test and mode!='parked' and not ((args.blocked_goal or block_injected) and mode=='no_safe_path'):
    raise RuntimeError('Parking test did not pass: '+mode)
