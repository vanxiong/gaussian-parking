"""CPU-only path search. It cannot block the LiDAR/GaussianFormer3D evidence worker."""
import json,sys,time
from pathlib import Path
import numpy as np
from parking_planner import EvidenceMap,Planner

print('PLANNER_READY',flush=True)
for line in sys.stdin:
    request={}
    try:
        request=json.loads(line);started=time.perf_counter()
        if request.get('test_delay'):time.sleep(float(request['test_delay']))
        capture=Path(request['capture'])
        with np.load(capture/'costmap.npz',allow_pickle=False) as saved:
            evidence=EvidenceMap()
            for name in ('free','obstacle','semantic','origin'):setattr(evidence,name,saved[name].copy())
            evidence.res=float(saved['res'])
        if request.get('road_layout'):
            from navigation import plan_to_destination
            path,planning=plan_to_destination(evidence,request['footprint'],request['start'],request['goal'],
                request['road_layout'],timeout=float(request.get('timeout',40.)),terminal_direction=request.get('terminal_direction'))
        else:
            planner=Planner(evidence,request['footprint'])
            path,planning=planner.plan(request['start'],request['goal'],timeout=float(request.get('timeout',20.)))
            if path is None and not planner.clearance(request['goal'])[0]:
                detail=planner.clearance_diagnostics(request['goal'])
                planning['clearance_diagnostics']=detail
                planning['goal_rejection']='obstacle' if detail['obstacle_cells'] else 'outside_map' if detail['out_of_bounds_cells'] else 'unobserved'
        response={'id':request['id'],'path':None if path is None else path.tolist(),'planning':planning,
                  'start':request['start'],'goal':request['goal'],'seconds':time.perf_counter()-started}
        (capture/f"planning_{request['id']:04d}.json").write_text(json.dumps(response,indent=2))
        print('PLANNER_RESULT '+json.dumps(response),flush=True)
    except Exception as error:
        import traceback;traceback.print_exc()
        print('PLANNER_RESULT '+json.dumps({'id':request.get('id'),'error':str(error)}),flush=True)
