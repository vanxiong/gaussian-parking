"""Project model voxel classes into the matching captured camera images."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def label_font(size):
    for path in ('C:/Windows/Fonts/arialbd.ttf', 'DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)

CLASSES = ('noise', 'barrier', 'bicycle', 'bus', 'car', 'construction vehicle',
           'motorcycle', 'pedestrian', 'traffic cone', 'trailer', 'truck',
           'road', 'other flat', 'sidewalk', 'terrain', 'manmade', 'vegetation')
COLORS = ((0,0,0), (255,120,60), (255,80,180), (255,210,60), (70,200,255),
          (220,150,50), (220,80,220), (255,70,70), (255,150,30), (160,120,255),
          (110,160,255), (80,190,100), (140,160,100), (180,130,180),
          (150,130,80), (190,160,255), (100,230,100))


def overlay_predictions(snapshots, metadata, labels, confidence, world_from_reference, compact_result=None):
    """Use projected voxels to place class names, without drawing the voxels."""
    selected = (labels > 0) & (labels < len(CLASSES)) & (confidence >= .35)
    indices = np.argwhere(selected)
    xyz = (indices + .5) * .5 + [-50., -50., -5.]
    reference_to_ego = np.linalg.inv(np.asarray(metadata['world_from_ego'])) @ world_from_reference
    ego = np.c_[xyz, np.ones(len(xyz))] @ reference_to_ego.T
    classes = labels[selected]
    cameras = {c['name']: c for c in metadata['cameras']}
    result = []
    for name, pixels in snapshots:
        image = Image.fromarray(pixels).convert('RGBA')
        camera = cameras[name]
        optical = ego @ np.linalg.inv(np.asarray(camera['ego_from_optical'])).T
        visible = optical[:, 2] > .5
        projected = optical[visible, :3] @ np.asarray(camera['intrinsic']).T
        uv = projected[:, :2] / projected[:, 2:3]
        ids = classes[visible]
        depth = optical[visible, 2]
        inside = (uv[:,0] >= 0) & (uv[:,0] < image.width) & (uv[:,1] >= 0) & (uv[:,1] < image.height)
        uv, ids, depth = uv[inside], ids[inside], depth[inside]
        # Z-buffer at small image bins keeps distant predictions behind nearer ones.
        occupied = set(); locations = {}
        for i in np.argsort(depth):
            x,y = uv[i].astype(int); key = (x//6, y//6)
            if key in occupied: continue
            occupied.add(key); cls = int(ids[i])
            locations.setdefault(cls, []).append((x,y))
        result.append((name,draw_labels(pixels,locations)))
        if compact_result is not None:
            compact_result.append((name,draw_labels(pixels,locations,1/3)))
    return result


def draw_labels(pixels,locations,label_scale=1.):
    image=Image.fromarray(pixels).convert('RGBA')
    draw = ImageDraw.Draw(image)
    font = label_font(max(6,round(max(18,round(image.width*12/220))*label_scale)))
    padding=max(1,round(4*label_scale))
    placed = []
    for cls, positions in locations.items():
        if len(positions) < 6: continue
        x,y = np.median(positions, axis=0).astype(int)
        text = CLASSES[cls]
        box = draw.textbbox((0,0),text,font=font,anchor='lt')
        width,height = box[2]+2*padding,box[3]+2*padding
        x = max(0,min(x-width//2,image.width-width))
        y = max(0,min(y,image.height-height))
        # Move overlapping labels to an available row within the image.
        for _ in range(image.height//height+1):
            rect = (x,y,x+width,y+height)
            if not any(rect[0]<b[2] and rect[2]>b[0] and rect[1]<b[3] and rect[3]>b[1] for b in placed):
                break
            y = (y+height+max(1,round(3*label_scale))) % max(1,image.height-height)
        else:
            continue
        placed.append(rect)
        draw.rounded_rectangle(rect,radius=max(1,round(5*label_scale)),fill=(10,15,25,255),outline=COLORS[cls],width=max(1,round(2*label_scale)))
        draw.text((x+padding,y+padding),text,font=font,anchor='lt',fill=COLORS[cls])
    return np.asarray(image.convert('RGB'))
