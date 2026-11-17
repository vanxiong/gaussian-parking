"""Inspect LiDAR/camera alignment and save an image overlay without model inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from prepare_capture import ORDER, rigid

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('capture', type=Path)
args = parser.parse_args()
metadata = json.loads((args.capture/'capture.json').read_text())
points = np.load(args.capture/metadata['lidar']['points'], allow_pickle=False)
points = points[np.isclose(points[:,4],0)]
ego_from_lidar = rigid(metadata['lidar']['ego_from_lidar'])
world_from_lidar = rigid(metadata['world_from_ego']) @ ego_from_lidar
world_xyz = points[:,:3] @ world_from_lidar[:3,:3].T + world_from_lidar[:3,3]
report = {'current_points': len(points), 'finite': bool(np.isfinite(points).all()),
          'world_xyz_min': world_xyz.min(0).tolist(), 'world_xyz_max': world_xyz.max(0).tolist(),
          'points_near_authored_ground_z0': int((np.abs(world_xyz[:,2])<.05).sum()), 'cameras': {}}
canvas = Image.new('RGB',(960,600),(18,18,18))
cameras = {c['name']:c for c in metadata['cameras']}
for index, name in enumerate(ORDER):
    camera = cameras[name]
    image = Image.open(args.capture/camera['image']).convert('RGB')
    draw = ImageDraw.Draw(image)
    optical_from_lidar = np.linalg.inv(rigid(camera['ego_from_optical'])) @ ego_from_lidar
    optical = points[:,:3] @ optical_from_lidar[:3,:3].T + optical_from_lidar[:3,3]
    optical = optical[optical[:,2]>.1]
    projected = optical @ np.asarray(camera['intrinsic']).T
    uv = projected[:,:2]/projected[:,2:3]
    valid = (uv[:,0]>=0)&(uv[:,0]<image.width)&(uv[:,1]>=0)&(uv[:,1]<image.height)
    uv, z = uv[valid], optical[valid,2]
    report['cameras'][name] = {'projected_lidar_hits': int(valid.sum())}
    for (u,v), depth in zip(uv[::3],z[::3]):
        red = int(np.clip(255-depth*6,0,255))
        draw.ellipse((u-1,v-1,u+1,v+1),fill=(red,255-red,80))
    image.thumbnail((320,270))
    x,y=(index%3)*320,(index//3)*300
    canvas.paste(image,(x,y+25))
    ImageDraw.Draw(canvas).text((x+6,y+6),name,fill='white')
canvas.save(args.capture/'lidar_camera_alignment.png')
(args.capture/'lidar_validation.json').write_text(json.dumps(report,indent=2))
assert report['finite'] and report['points_near_authored_ground_z0']>100
assert all(record['projected_lidar_hits']>20 for record in report['cameras'].values())
print(json.dumps(report,indent=2))
