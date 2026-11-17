"""Build and render the CMPE294 static parking perception starter scene."""
import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--gui', action='store_true')
args, _ = parser.parse_known_args()
from isaacsim import SimulationApp
app = SimulationApp({'headless': not args.gui, 'renderer': 'RayTracedLighting', 'width': 1600, 'height': 1000})
import omni.usd
import omni.replicator.core as rep
from pxr import Gf, UsdGeom, UsdLux, UsdShade, UsdPhysics
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, '/World')
stage.SetDefaultPrim(world.GetPrim())
UsdPhysics.Scene.Define(stage, '/World/Physics').CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))

def material(name, color, roughness=.6, metallic=0):
    mat = UsdShade.Material.Define(stage, '/World/Materials/' + name)
    shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + '/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    from pxr import Sdf
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return mat

mats = {k: material(k, c, r, m) for k,c,r,m in [
    ('asphalt', (.095,.115,.14), .95,0), ('concrete',(.52,.55,.56),.9,0),
    ('white',(.85,.88,.83),.7,0), ('yellow',(.95,.64,.055),.65,0),
    ('blue',(.025,.28,.53),.28,.35), ('red',(.56,.07,.045),.3,.3),
    ('silver',(.55,.62,.67),.3,.55), ('teal',(.045,.35,.32),.35,.3),
    ('glass',(.055,.10,.15),.12,.45), ('rubber',(.015,.02,.025),.85,0),
    ('chrome',(.55,.6,.64),.23,.8), ('orange',(.95,.22,.025),.7,0),
    ('grass',(.15,.28,.12),.95,0), ('leaf',(.12,.30,.16),.9,0),
    ('bark',(.24,.15,.075),.9,0), ('wall',(.35,.40,.46),.85,0)]}

def finish(obj, pos, scale, mat, collision=False, rot=None):
    xf = UsdGeom.Xformable(obj.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if rot: xf.AddRotateXYZOp().Set(Gf.Vec3f(*rot))
    xf.AddScaleOp().Set(Gf.Vec3f(*scale))
    UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(mats[mat])
    if collision: UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
    return obj

def box(path, pos, size, mat, collision=False):
    obj = UsdGeom.Cube.Define(stage, path)
    obj.CreateSizeAttr(1)
    return finish(obj, pos, size, mat, collision)

def cylinder(path, pos, radius, height, mat, rot=None):
    obj = UsdGeom.Cylinder.Define(stage, path)
    obj.CreateRadiusAttr(radius); obj.CreateHeightAttr(height)
    return finish(obj, pos, (1,1,1), mat, rot=rot)

box('/World/Ground', (0,0,-.15), (70,60,.2), 'grass', True)
box('/World/Lot', (0,0,-.025), (42,30,.15), 'asphalt', True)
for y in [-15,15]: box('/World/Curbs/Y'+str(y).replace('-','N'), (0,y,.13),(42,.3,.32),'concrete',True)
for x in [-21,21]: box('/World/Curbs/X'+str(x).replace('-','N'), (x,0,.13),(.3,30,.32),'concrete',True)
# Two rows of 2.8 x 5.2 m parking bays, a 9 m aisle, and wheel stops.
for row,y in enumerate([-7.3,7.3]):
    for i in range(13):
        x=-18.2+i*2.8
        box(f'/World/Markings/R{row}_L{i}',(x,y,.058),(.09,5.2,.012),'white')
    for i in range(12):
        x=-16.8+i*2.8
        box(f'/World/WheelStops/R{row}_{i}',(x,y+(-2 if y<0 else 2),.16),(1.6,.22,.22),'concrete',True)
    box(f'/World/Markings/Back{row}',(-1.4,y+(-2.6 if y<0 else 2.6),.058),(33.6,.09,.012),'white')
for i in range(12): box(f'/World/Markings/Center{i}',(-18+i*3.2,0,.058),(1.5,.10,.012),'yellow')

def car(path, position, yaw, color):
    parent=UsdGeom.Xform.Define(stage,path)
    xf=UsdGeom.Xformable(parent); xf.AddTranslateOp().Set(Gf.Vec3d(*position)); xf.AddRotateZOp().Set(yaw)
    box(path+'/Body',(0,0,.62),(4.3,1.82,.70),color,True)
    box(path+'/Hood',(1.35,0,.97),(1.3,1.78,.16),color)
    # Tapered cabin mesh with windscreen and roof.
    points=[(-1.25,-.85,.95),(1.05,-.85,.95),(1.05,.85,.95),(-1.25,.85,.95),
            (-.75,-.67,1.62),(.45,-.67,1.62),(.45,.67,1.62),(-.75,.67,1.62)]
    mesh=UsdGeom.Mesh.Define(stage,path+'/Cabin'); mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4]*6)
    mesh.CreateFaceVertexIndicesAttr([0,3,2,1,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7,4,5,6,7])
    mesh.CreateSubdivisionSchemeAttr('none')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mats['glass'])
    box(path+'/Roof',(-.15,0,1.64),(1.25,1.37,.07),color)
    for x in [-1.35,1.35]:
        for y in [-.94,.94]:
            suffix=('F' if x>0 else 'B')+('L' if y>0 else 'R')
            cylinder(path+'/Tire'+suffix,(x,y,.39),.38,.23,'rubber',(90,0,0))
            cylinder(path+'/Rim'+suffix,(x,y+( .125 if y>0 else -.125),.39),.23,.018,'chrome',(90,0,0))
    for y in [-.57,.57]:
        box(path+'/Headlight'+('L' if y>0 else 'R'),(2.16,y,.82),(.03,.4,.16),'white')
        box(path+'/Tail'+('L' if y>0 else 'R'),(-2.16,y,.82),(.03,.4,.16),'red')
    box(path+'/Grille',(2.16,0,.52),(.035,.85,.16),'rubber')
    for y in [-1,1]: box(path+'/Mirror'+('L' if y>0 else 'R'),(.55,y,1.15),(.24,.17,.12),color)
    return parent

ego=car('/World/EgoVehicle',(-5,-2,0),0,'blue')
for i,(x,y,c) in enumerate([(-16.8,7.3,'silver'),(-11.2,7.3,'red'),(-5.6,7.3,'teal'),(2.8,7.3,'silver'),(8.4,7.3,'blue'),(14,7.3,'red'),(-14,-7.3,'teal'),(0,-7.3,'silver'),(8.4,-7.3,'red'),(14,-7.3,'silver')]):
    car(f'/World/ParkedVehicles/Car{i}',(x,y,0),90 if y>0 else -90,c)
for i,(x,y) in enumerate([(1,-2),(2.2,-2),(3.4,-2),(4.6,-2),(6,-1)]):
    path=f'/World/Cones/C{i}'
    box(path+'/Base',(x,y,.09),(.48,.48,.08),'rubber',True)
    cone=UsdGeom.Cone.Define(stage,path+'/Cone'); cone.CreateRadiusAttr(.19); cone.CreateHeightAttr(.65)
    finish(cone,(x,y,.43),(1,1,1),'orange',True)
    cylinder(path+'/Band',(x,y,.42),.12,.12,'white')
for i,x in enumerate([-17,-8,1,10,18]):
    box(f'/World/Planters/P{i}',(x,12,.24),(2.6,2,.45),'concrete',True)
    box(f'/World/Planters/Soil{i}',(x,12,.48),(2.3,1.7,.05),'bark')
    cylinder(f'/World/Trees/Trunk{i}',(x,12,1.9),.16,2.8,'bark')
    sphere=UsdGeom.Sphere.Define(stage,f'/World/Trees/Crown{i}'); sphere.CreateRadiusAttr(1)
    finish(sphere,(x,12,3.4),(1.3,1.25,1.7),'leaf')
box('/World/Building/Main',(0,22,3),(32,8,6),'wall',True)
for i in range(8): box(f'/World/Building/Window{i}',(-14+i*4,17.98,3.5),(2.8,.035,2),'glass')
box('/World/Building/Canopy',(0,17.5,4.9),(33,1.5,.2),'chrome')
for i,x in enumerate([-18,18]):
    cylinder(f'/World/Lamp/Pole{i}',(x,11,3),.065,6,'chrome')
    box(f'/World/Lamp/Head{i}',(x,10.7,6),(.7,.9,.12),'chrome')
dome=UsdLux.DomeLight.Define(stage,'/World/Lighting/Sky'); dome.CreateIntensityAttr(650)
dome.CreateColorAttr(Gf.Vec3f(.79,.87,1))
sun=UsdLux.DistantLight.Define(stage,'/World/Lighting/Sun'); sun.CreateIntensityAttr(2200)
sun.CreateAngleAttr(1.2); UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-35,-25,25))

def camera(path, eye, target, hfov=78, resolution=(960,540)):
    cam=UsdGeom.Camera.Define(stage,path)
    m=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse()
    UsdGeom.Xformable(cam).AddTransformOp().Set(m)
    aperture=20.955
    focal=aperture/(2*math.tan(math.radians(hfov)/2))
    cam.CreateHorizontalApertureAttr(aperture); cam.CreateVerticalApertureAttr(aperture*resolution[1]/resolution[0])
    cam.CreateFocalLengthAttr(focal); cam.CreateClippingRangeAttr(Gf.Vec2f(.05,150))
    fx=resolution[0]/(2*math.tan(math.radians(hfov)/2))
    # USD stores row-vector matrices. Export column-vector ROS optical convention.
    optical=np.array(m).T @ np.diag([1,-1,-1,1])
    return {'path':path,'resolution':list(resolution),'K':[[fx,0,resolution[0]/2],[0,fx,resolution[1]/2],[0,0,1]],'T_parent_from_optical':optical.tolist(),'hfov_degrees':hfov}

rig=[]
for name,yaw,pos in [('CAM_FRONT',0,(1.3,0,1.75)),('CAM_FRONT_LEFT',55,(.8,.75,1.75)),('CAM_FRONT_RIGHT',-55,(.8,-.75,1.75)),('CAM_BACK',180,(-1.5,0,1.6)),('CAM_BACK_LEFT',125,(-.9,.75,1.65)),('CAM_BACK_RIGHT',-125,(-.9,-.75,1.65))]:
    a=math.radians(yaw); target=(pos[0]+10*math.cos(a),pos[1]+10*math.sin(a),pos[2]-.5)
    record=camera('/World/EgoVehicle/Cameras/'+name,pos,target)
    record['name']=name; rig.append(record)
camera('/World/Overview',(27,-33,27),(-1,1,0),55,(1600,1000))
stage.GetRootLayer().Save() if stage.GetRootLayer().realPath else None
stage.GetRootLayer().Export(str(OUT/'parking_scene.usda'))
(OUT/'camera_calibration.json').write_text(json.dumps({'units':'meters','ego_axes':'X forward, Y left, Z up','camera_axes':'ROS optical: X right, Y down, Z forward','calibration_status':'approximate starter rig; not nuScenes calibrated','cameras':rig},indent=2))
print('SCENE_SAVED',flush=True)
products=[]
for name,path,resolution in [('overview','/World/Overview',(1600,1000))]+[(r['name'],r['path'],tuple(r['resolution'])) for r in rig]:
    rp=rep.create.render_product(path,resolution)
    annot=rep.AnnotatorRegistry.get_annotator('rgb'); annot.attach([rp])
    products.append((name,annot,rp))
for _ in range(60): app.update()
rep.orchestrator.step(rt_subframes=16)
stats={}
for name,annot,rp in products:
    data=np.asarray(annot.get_data())
    if data.size==0: raise RuntimeError('Empty camera image: '+name)
    rgb=data[:,:,:3].astype(np.uint8)
    Image.fromarray(rgb).save(OUT/(name+'.png'))
    stats[name]={'shape':list(rgb.shape),'mean':float(rgb.mean()),'std':float(rgb.std())}
    if rgb.std()<5: raise RuntimeError('Flat camera image: '+name)
sheet=Image.new('RGB',(1440,620),(22,27,33)); draw=ImageDraw.Draw(sheet)
for i,r in enumerate(rig):
    x=(i%3)*480; y=(i//3)*310
    sheet.paste(Image.open(OUT/(r['name']+'.png')).resize((480,270)),(x,y+35))
    draw.text((x+12,y+10),r['name'],fill='white')
sheet.save(OUT/'surround_views.png')
(OUT/'validation.json').write_text(json.dumps(stats,indent=2))
print('CAPTURE_VALIDATED',json.dumps(stats),flush=True)
if args.gui:
    from isaacsim.core.utils.viewports import set_camera_view
    set_camera_view(eye=np.array([27,-33,27]),target=np.array([-1,1,0]))
    while app.is_running(): app.update()
app.close()
