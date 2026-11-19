"""Export visible USD geometry independently of sensors/predictions."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from collections import Counter
APP=None
try:
    from pxr import Usd,UsdGeom
except ModuleNotFoundError:
    from isaacsim import SimulationApp
    APP=SimulationApp({'headless':True,'width':320,'height':240})
    from pxr import Usd,UsdGeom

def primitive_points(prim):
    kind=prim.GetTypeName()
    if kind=='Mesh':
        mesh=UsdGeom.Mesh(prim)
        points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
        counts=mesh.GetFaceVertexCountsAttr().Get();indices=mesh.GetFaceVertexIndicesAttr().Get();offset=0;edges=Counter()
        for count in counts:
            face=list(indices[offset:offset+count]);offset+=count
            for a,b in zip(face,face[1:]+face[:1]):edges[tuple(sorted((a,b)))]+=1
            vertices=points[face];normal=np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0]);norm=np.linalg.norm(normal)
            if norm<1e-10:raise ValueError('Degenerate mesh face: '+str(prim.GetPath()))
            distances=(points-vertices[0])@(normal/norm)
            if distances.min() < -1e-5 and distances.max()>1e-5:raise ValueError('Nonconvex mesh unsupported: '+str(prim.GetPath()))
            if np.abs((vertices-vertices[0])@(normal/norm)).max()>1e-5:raise ValueError('Nonplanar mesh face unsupported')
        if any(count!=2 for count in edges.values()):raise ValueError('Open mesh cannot define a filled solid: '+str(prim.GetPath()))
        return points
    if kind=='Cube':
        size=UsdGeom.Cube(prim).GetSizeAttr().Get()/2
        return np.array([[x,y,z] for x in [-size,size] for y in [-size,size] for z in [-size,size]])
    if kind in ['Cylinder','Cone']:
        obj=getattr(UsdGeom,kind)(prim);r=obj.GetRadiusAttr().Get();h=obj.GetHeightAttr().Get()/2
        angles=np.arange(32)*2*math.pi/32
        rings=[np.c_[r*np.cos(angles),r*np.sin(angles),np.full(32,-h)]]
        rings.append(np.array([[0.,0.,h]]) if kind=='Cone' else np.c_[r*np.cos(angles),r*np.sin(angles),np.full(32,h)])
        points=np.concatenate(rings)
        axis=obj.GetAxisAttr().Get()
        if axis=='X':points=points[:,[2,0,1]]
        elif axis=='Y':points=points[:,[1,2,0]]
        return points
    if kind=='Sphere':
        r=UsdGeom.Sphere(prim).GetRadiusAttr().Get()
        return np.array([[r*math.sin(p)*math.cos(t),r*math.sin(p)*math.sin(t),r*math.cos(p)]
                         for p in np.linspace(0,math.pi,17) for t in np.arange(32)*math.pi/16])
    raise ValueError('Unsupported visible geometry: '+str(prim.GetPath())+' '+kind)

def export(scene,output):
    stage=Usd.Stage.Open(str(scene));assert stage
    if UsdGeom.GetStageUpAxis(stage)!='Z' or UsdGeom.GetStageMetersPerUnit(stage)!=1:raise ValueError('Expected Z-up meter scene')
    cache=UsdGeom.XformCache();objects=[];excluded=[];proxy_paths=[];visual_paths=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):continue
        path=str(prim.GetPath());imageable=UsdGeom.Imageable(prim)
        ancestor=prim;visual_only=False
        while ancestor and not ancestor.IsPseudoRoot():
            if ancestor.GetAttribute('occupancy:excludeFromGroundTruth').Get():visual_only=True;break
            ancestor=ancestor.GetParent()
        if visual_only:
            excluded.append(path);visual_paths.append(path);continue
        proxy=bool(prim.GetAttribute('occupancy:groundTruthProxy').Get())
        if path.startswith(('/World/PhysXCar','/World/EgoVehicle','/World/DrivingDebug','/World/ROSDrivingDebug')) or imageable.ComputeVisibility()=='invisible' or (imageable.ComputePurpose() in ['guide','proxy'] and not proxy):
            excluded.append(path);continue
        if proxy:proxy_paths.append(path)
        points=primitive_points(prim)
        transform=np.asarray(cache.GetLocalToWorldTransform(prim)).T
        world=points@transform[:3,:3].T+transform[:3,3]
        objects.append({'path':path,'type':prim.GetTypeName(),'points_world':world.tolist()})
    ground=next(obj for obj in objects if obj['path']=='/World/Ground')
    bounds=np.array(ground['points_world'])
    report={'schema':'gf3d.usd_geometry.v1','scene':str(scene.resolve()),
            'scene_sha256':hashlib.sha256(scene.read_bytes()).hexdigest(),'objects':objects,
            'excluded':excluded,'known_xy_bounds':[bounds[:,:2].min(0).tolist(),bounds[:,:2].max(0).tolist()],
            'curves':'32 angular segments; spheres 16 latitude bands',
            'mesh_policy':'Benchmark accepts convex mesh solids only; evaluator rejects nonconvex topology.'}
    report['appearance_reference']={'policy':'shared_simple_car_proxy_v1' if proxy_paths else 'rendered_geometry',
        'proxy_paths':proxy_paths,'excluded_detailed_visual_paths':visual_paths,
        'description':'Selected original car shapes retained as explicit occupancy proxies; detailed visual meshes excluded.' if proxy_paths else 'Visible rendered geometry.'}
    ego=stage.GetPrimAtPath('/World/PhysXCar')
    if ego:report['world_from_ego']=np.asarray(cache.GetLocalToWorldTransform(ego)).T.tolist()
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report))
    print(json.dumps({'geometry_objects':len(objects),'excluded':len(excluded),'output':str(output)}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('scene',type=Path,nargs='?');parser.add_argument('output',type=Path,nargs='?')
    parser.add_argument('--batch',type=Path);parser.add_argument('--no-ros-env',action='store_true');args=parser.parse_args()
    if args.batch:
        for job in json.loads(args.batch.read_text()):export(Path(job['scene']),Path(job['output']))
    elif args.scene and args.output:export(args.scene,args.output)
    else:parser.error('Provide scene/output or --batch')

if __name__=='__main__':
    try:main()
    finally:
        if APP is not None:APP.close()
