"""Check LiDAR clearance and parking search on the validated frozen-model capture."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simulation'))
from parking_planner import EvidenceMap,Planner

latest=json.loads((ROOT/'reports/latest_prediction.json').read_text())
capture=ROOT/latest['capture'];prediction=ROOT/latest['prediction']
report=json.loads((prediction/'prediction.json').read_text())
labels=np.load(prediction/'raw_labels.npy',allow_pickle=False)
confidence=np.load(prediction/'confidence.npy',allow_pickle=False)
evidence=EvidenceMap()
metrics=evidence.update(capture,labels,confidence,np.asarray(report['world_from_lidar']),[4.3,1.82],.3)
planner=Planner(evidence,[4.3,1.82])
start=[-6.35,-2.,0.];goal=[-2.8,5.55,np.pi/2]
output={'metrics':metrics,'start':planner.clearance_diagnostics(start),'goal':planner.clearance_diagnostics(goal)}
path,planning=planner.plan(start,goal,timeout=20.)
output['planning']=planning
output['path_found']=path is not None
(ROOT/'reports/lidar_parking_map.json').write_text(json.dumps(output,indent=2))
print(json.dumps({key:value for key,value in output.items() if key not in ('start','goal')},indent=2))
print('Start:',{k:v for k,v in output['start'].items() if not k.endswith('_xy')})
print('Goal:',{k:v for k,v in output['goal'].items() if not k.endswith('_xy')})
if path is None:raise SystemExit('LiDAR map cannot support this parking route yet')
