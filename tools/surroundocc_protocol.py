"""Author-compatible semantic/geometric metrics and confined dataset paths."""
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
CLASSES=['barrier','bicycle','bus','car','construction_vehicle','motorcycle','pedestrian',
         'traffic_cone','trailer','truck','driveable_surface','other_flat','sidewalk','terrain','manmade','vegetation']
EXPECTED_SAMPLES=6019

def confined(path,root=DATA):
    root=Path(root).resolve();path=Path(path).resolve();path.relative_to(root);return path

def resolve_sensor(path,root):
    """Metadata may contain the author's absolute prefix; preserve its sensor suffix."""
    value=str(path).replace('\\','/')
    for marker in ['samples/','sweeps/']:
        position=value.find(marker)
        if position>=0:return confined(Path(root)/value[position:],root)
    return confined(Path(root)/value,root)

def sparse_labels(values):
    values=np.asarray(values)
    if values.ndim!=2 or values.shape[1]!=4 or not np.isfinite(values).all() or not np.equal(values,np.floor(values)).all():raise ValueError('Expected integer Nx4 SurroundOcc annotations')
    if (values[:,:3]<0).any() or (values[:,:3]>=np.array([200,200,16])).any() or (values[:,3]<0).any() or (values[:,3]>16).any():raise ValueError('Invalid occupancy index or semantic class')
    result=np.full((200,200,16),17,np.int64);indices=values[:,:3].astype(int)
    if len(np.unique(indices,axis=0))!=len(indices):raise ValueError('Duplicate occupancy coordinates')
    result[tuple(indices.T)]=values[:,3].astype(int)
    return result

def confusion_matrix(prediction,target,ignore_zero=False):
    p=np.asarray(prediction).ravel();t=np.asarray(target).ravel()
    if p.shape!=t.shape or ((p<0)|(p>17)|(t<0)|(t>17)).any():raise ValueError('Invalid labels or shape')
    if ignore_zero:p,t=p[t!=0],t[t!=0]
    return np.bincount(t.astype(np.int64)*18+p.astype(np.int64),minlength=324).reshape(18,18)

def summarize(matrix):
    matrix=np.asarray(matrix,dtype=np.int64);seen=matrix.sum(1);positive=matrix.sum(0);correct=matrix.diagonal()
    per_class=[]
    for i,name in enumerate(CLASSES,1):
        # Match MeanIoU._after_epoch exactly, including its absent-GT-class convention.
        iou=float(correct[i]/(seen[i]+positive[i]-correct[i])) if seen[i] else 1.
        per_class.append({'label':i,'name':name,'iou':iou,'gt_voxels':int(seen[i]),'predicted_voxels':int(positive[i]),'true_positive_voxels':int(correct[i])})
    tp=int(matrix[:17,:17].sum());fp=int(matrix[17,:17].sum());fn=int(matrix[:17,17].sum())
    return {'occupied_iou':tp/(tp+fp+fn) if tp+fp+fn else None,
            'semantic_miou':float(np.mean([c['iou'] for c in per_class])),
            'per_class':per_class,'occupied_counts':{'tp':tp,'fp':fp,'fn':fn},'voxels':int(matrix.sum())}
