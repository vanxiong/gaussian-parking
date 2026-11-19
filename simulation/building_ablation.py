"""Remove the whole building subtree for a recorded, reversible scene variant."""
from pxr import Usd,UsdGeom,UsdPhysics

def apply_building_presence(stage,presence,colliders):
    if presence not in ('present','absent'):raise ValueError('Unknown building presence')
    path='/World/Building';building=stage.GetPrimAtPath(path)
    if not building:raise ValueError('Parking-lot building subtree is missing')
    shapes=[str(p.GetPath()) for p in Usd.PrimRange(building) if p.IsA(UsdGeom.Gprim)]
    collisions=[str(p.GetPath()) for p in Usd.PrimRange(building) if p.HasAPI(UsdPhysics.CollisionAPI)]
    record={'presence':presence,'factor':'building_presence','root_path':path,
            'building_shape_paths':shapes,'building_collision_paths':collisions,
            'removal_scope':'Entire building subtree including windows and canopy; rendered, collision/LiDAR and reference geometry removed together',
            'removed_shape_count':len(shapes) if presence=='absent' else 0}
    if presence=='absent':
        stage.RemovePrim(path)
        colliders[:]=[item for item in colliders if not (item[0]==path or item[0].startswith(path+'/'))]
    return record
