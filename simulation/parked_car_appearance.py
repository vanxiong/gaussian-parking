"""Toggle all parked sedans while holding occupancy references fixed."""
import hashlib,json
from pathlib import Path
import numpy as np
from pxr import Gf,Sdf,Usd,UsdGeom,UsdShade

ROOT=Path(__file__).resolve().parents[1]
CAR_IDS=tuple(range(10))
PROXY_ATTR='occupancy:groundTruthProxy'
VISUAL_ATTR='occupancy:excludeFromGroundTruth'

def apply_appearance(stage,variant):
    if variant not in ('simple','realistic'):raise ValueError('Unknown parked-car appearance')
    record={'variant':variant,'factor':'parked_car_visual_appearance','selected_car_ids':list(CAR_IDS),
        'reference_geometry':'Original simple-car geometry in both variants; detailed mesh excluded from target',
        'collision_and_lidar':'Original collision shapes unchanged',
        'ground_truth_policy':'shared_simple_car_proxy_v1','cars':[]}
    if variant=='simple':return record
    asset=ROOT/'assets/parked_sedan/sedan.usda';metadata=json.loads((asset.parent/'asset.json').read_text())
    digest=hashlib.sha256(asset.read_bytes()).hexdigest()
    if digest!=metadata['usd_sha256']:raise ValueError('Detailed sedan differs from pinned asset metadata')
    record.update(asset_sha256=digest,source_sha256=metadata['source_sha256'],source_url=metadata['source_url'],asset_license=metadata['license'])
    for number in CAR_IDS:
        path=f'/World/ParkedVehicles/Car{number}';car=stage.GetPrimAtPath(path)
        if not car:raise ValueError('Selected parked car missing: '+path)
        original=[p for p in Usd.PrimRange(car) if p.IsA(UsdGeom.Gprim)]
        cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_])
        bounds=cache.ComputeUntransformedBound(car).ComputeAlignedRange();lo=np.array(bounds.GetMin());hi=np.array(bounds.GetMax())
        world_before=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(car)).T
        body_material,_=UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path+'/Body')).ComputeBoundMaterial()
        color=UsdShade.Shader(stage.GetPrimAtPath(str(body_material.GetPath())+'/Shader')).GetInput('diffuseColor').Get()
        for prim in original:
            prim.CreateAttribute(PROXY_ATTR,Sdf.ValueTypeNames.Bool,custom=True).Set(True)
            UsdGeom.Imageable(prim).CreatePurposeAttr().Set(UsdGeom.Tokens.proxy)
        visual=UsdGeom.Xform.Define(stage,path+'/DetailedAppearance')
        visual.GetPrim().CreateAttribute(VISUAL_ATTR,Sdf.ValueTypeNames.Bool,custom=True).Set(True)
        visual.AddTranslateOp().Set(Gf.Vec3d(float((lo[0]+hi[0])/2),float((lo[1]+hi[1])/2),float(lo[2])))
        scale=(hi-lo)/np.asarray(metadata['normalized_size_xyz'])
        visual.AddScaleOp().Set(Gf.Vec3f(*scale))
        referenced=stage.DefinePrim(path+'/DetailedAppearance/Asset','Xform')
        referenced.GetReferences().AddReference(str(asset))
        # Preserve each car's baseline paint color rather than adding a color factor.
        shader=UsdShade.Shader(stage.GetPrimAtPath(str(referenced.GetPath())+'/Looks/Mat2/Shader'))
        if not shader:raise RuntimeError('Detailed asset failed to compose')
        shader.GetInput('diffuseColor').Set(color)
        record['cars'].append({'path':path,'world_from_car':world_before.tolist(),'reference_bounds_local':[lo.tolist(),hi.tolist()],
            'visual_scale_xyz':scale.tolist(),'paint_rgb':list(color),'proxy_prim_count':len(original)})
    return record
