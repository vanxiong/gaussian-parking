"""Verify the appearance ablation preserves geometry/collisions and render A/B."""
import argparse,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('scene',type=Path);p.add_argument('--output',type=Path,required=True);a,_=p.parse_known_args();a.output.mkdir(parents=True,exist_ok=True)
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'simulation'))
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'width':1200,'height':700})
try:
    import omni.usd,omni.replicator.core as rep
    from pxr import Usd,UsdGeom,UsdPhysics,Gf
    from PIL import Image,ImageDraw
    from parked_car_appearance import apply_appearance,CAR_IDS
    from export_eval_geometry import export
    omni.usd.get_context().open_stage(str(a.scene));stage=omni.usd.get_context().get_stage()
    def collision_state():
        return {str(p.GetPath()):{attr.GetName():str(attr.Get()) for attr in p.GetAttributes() if attr.GetName().startswith(('physics:','physx'))} for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)}
    collision_before=collision_state()
    def ego_state():
        return {str(p.GetPath()):{attr.GetName():str(attr.Get()) for attr in p.GetAttributes()} for p in Usd.PrimRange(stage.GetPrimAtPath('/World/PhysXCar'))}
    ego_before=ego_state()
    stage.GetRootLayer().Export(str(a.output/'simple-scene.usda'));export(a.output/'simple-scene.usda',a.output/'simple-geometry.json')
    camera=UsdGeom.Camera.Define(stage,'/World/AblationPreviewCamera')
    camera.CreateHorizontalApertureAttr(20.955);camera.CreateVerticalApertureAttr(20.955*540/960);camera.CreateFocalLengthAttr(24.)
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(6,-13,4),Gf.Vec3d(0,-7.3,.8),Gf.Vec3d(0,0,1)).GetInverse())
    product=rep.create.render_product(str(camera.GetPath()),(960,540));rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product])
    def capture(name):
        rep.orchestrator.step(delta_time=0.,rt_subframes=16,pause_timeline=True)
        data=np.asarray(rgb.get_data())[:,:,:3].astype(np.uint8)
        if data.shape!=(540,960,3) or data.std()<2:raise RuntimeError('Invalid preview render')
        im=Image.fromarray(data);im.save(a.output/(name+'.png'));return im
    baseline=capture('simple')
    record=apply_appearance(stage,'realistic')
    detailed=capture('realistic')
    stage.GetRootLayer().Export(str(a.output/'realistic-scene.usda'));export(a.output/'realistic-scene.usda',a.output/'realistic-geometry.json')
    simple=json.loads((a.output/'simple-geometry.json').read_text());real=json.loads((a.output/'realistic-geometry.json').read_text())
    if simple['objects']!=real['objects']:raise AssertionError('Reference objects changed between variants')
    if simple['known_xy_bounds']!=real['known_xy_bounds']:raise AssertionError('Evaluation bounds changed')
    if collision_before!=collision_state():raise AssertionError('Collision properties changed')
    if len(real['appearance_reference']['excluded_detailed_visual_paths'])!=len(CAR_IDS)*73:raise AssertionError('Expected 73 visual meshes per selected car')
    if ego_before!=ego_state():raise AssertionError('Driving car changed')
    for i in CAR_IDS:
        if not stage.GetPrimAtPath(f'/World/ParkedVehicles/Car{i}/DetailedAppearance/Asset/Part000'):raise AssertionError('Parked car missing detailed model')
    # The nonselected cars retain their default purposes and no visual children.
    for i in set(range(10))-set(CAR_IDS):
        car=stage.GetPrimAtPath(f'/World/ParkedVehicles/Car{i}')
        if any(p.GetAttribute('occupancy:groundTruthProxy').Get() for p in Usd.PrimRange(car)):raise AssertionError('Nonselected car changed')
    pair=Image.new('RGB',(1920,580),'#18212e');pair.paste(baseline,(0,40));pair.paste(detailed,(960,40))
    draw=ImageDraw.Draw(pair);draw.text((20,12),'SIMPLE (baseline)',fill='white');draw.text((980,12),f'REALISTIC ({len(CAR_IDS)} parked cars)',fill='white');pair.save(a.output/'appearance-comparison.png')
    result={'passed':True,'reference_objects_identical':True,'collision_properties_identical':True,'unselected_cars_unchanged':True,
        'driving_car_unchanged':True,'detailed_parked_car_count':len(CAR_IDS),
        'reference_object_count':len(simple['objects']),'detailed_visual_mesh_count':len(real['appearance_reference']['excluded_detailed_visual_paths']),
        'proxy_prim_count':len(real['appearance_reference']['proxy_paths']),'appearance_ablation':record}
    (a.output/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
finally:app.close()
