"""Read saved inputs; isolate coverage and random anchor assignment without changing deployment."""
import argparse,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/stop_prediction_diagnostic'
OUT.mkdir(parents=True,exist_ok=True)

def coverage(points):
    low=np.array([-50.,-50.,-5.]);high=np.array([50.,50.,3.])
    p=points[((points[:,:3]>=low)&(points[:,:3]<high)).all(1)]
    n=len(np.unique(np.floor((p[:,:3]-low)/[.075,.075,.2]).astype(np.int32),axis=0))
    return {'points':len(points),'unique_voxels':n,'unsupported_anchors':max(0,25600-n)}

def inventory():
    run=RUN
    rows=[];prev=None
    for f in sorted(run.glob('frame_*')):
        if not (f/'raw_labels.npy').exists():continue
        m=json.loads((f/'capture.json').read_text());p=np.load(f/m['lidar']['points'])
        t=m['simulation_timestamp_seconds'];pose=np.array(m['world_from_ego'])[:3,3]
        speed=None if prev is None else float(np.linalg.norm(pose[:2]-prev[1][:2])/max(t-prev[0],1e-8))
        row={'frame':f.name,'time':t,'speed_from_previous_capture':speed,'sweeps':m['lidar']['sweeps'],**coverage(p),'current':coverage(p[np.isclose(p[:,4],0)])}
        rows.append(row);prev=(t,pose)
    (OUT/'coverage.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))
    return rows

def replay():
    import torch,cv2
    from run_gaussianformer3d import build_model,build_queries,infer_arrays
    from prepare_capture import prepare
    from occupancy_evaluation import confusion,metrics
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.manual_seed(42);np.random.seed(42)
    model,config,_=build_model(anchor_assignment=ANCHOR_ASSIGNMENT);queries=build_queries(config)
    counts=[]
    def hook(module,args,kwargs):
        coords=kwargs.get('coors_batch')
        if coords is None and len(args)>2:coords=args[2]
        if coords is not None:counts.append(int((coords[:,0]==0).sum()))
    model.lifter.register_forward_pre_hook(hook,with_kwargs=True)
    rows=json.loads((OUT/'coverage.json').read_text());moving=[r for r in rows if (r['speed_from_previous_capture'] or 0)>.3]
    chosen=[('start',rows[0]['frame']),('moving',moving[len(moving)//2]['frame']),('end',rows[-1]['frame'])]
    cases=[(name,RUN/frame,None) for name,frame in chosen]
    report=REPORT
    manifest=json.loads((report/'manifest.json').read_text())
    cases.append(('stationary_scored',Path(manifest['samples'][0]['capture']),report/'frame_001-ground-truth.npz'))
    records=[]
    for name,capture,truthpath in cases:
        arrays=prepare(capture);truth=np.load(truthpath) if truthpath else None
        metadata=json.loads((capture/'capture.json').read_text())
        transform=np.asarray(metadata['world_from_ego'])@np.asarray(metadata['lidar']['ego_from_lidar'])
        local=(np.indices((200,200,16)).reshape(3,-1).T+.5)*.5+[-50.,-50.,-5.]
        world_z=(local@transform[:3,:3].T+transform[:3,3])[:,2].reshape(200,200,16)
        for variant in ['history','current_only']:
            inputs=dict(arrays)
            if variant=='current_only':inputs['points']=arrays['points'][np.isclose(arrays['points'][:,4],0)]
            reference=None;reference_display=None
            for seed in [42,42,43,44,45]:
                torch.manual_seed(seed);np.random.seed(seed);counts.clear()
                raw,conf,stats=infer_arrays(model,config,inputs,queries)
                display=(raw>0)&(~np.isin(raw,[11,12,13,14,17]))&(conf>=.35)&(world_z>.3)&(world_z<2.5)
                if reference is None:reference=raw.copy();reference_display=display.copy()
                union=np.count_nonzero(display|reference_display)
                row={'case':name,'capture':str(capture),'variant':variant,'seed':seed,'actual_lifter_voxels':counts[-1] if counts else None,
                     'coverage':coverage(inputs['points']),'label_agreement_to_first':float((raw==reference).mean()),
                     'display_jaccard_to_first':float(np.count_nonzero(display&reference_display)/union) if union else 1.,**stats}
                if truth is not None:
                    row['raw_score']=metrics(confusion(raw!=17,truth['occupied'],truth['obstacles_above_ground']))
                    row['display_score']=metrics(confusion(display,truth['occupied'],truth['obstacles_above_ground']))
                records.append(row)
                (OUT/'replays.json').write_text(json.dumps(records,indent=2))
                print(json.dumps(row),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--report',type=Path,help='Stationary evaluation directory, required for --replay')
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--replay',action='store_true')
    parser.add_argument('--anchor-assignment',choices=['random','deterministic'],required=True)
    args=parser.parse_args()
    if args.replay and args.report is None:parser.error('--report is required for --replay')
    RUN=args.run.resolve();REPORT=args.report.resolve() if args.report else None
    OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
    ANCHOR_ASSIGNMENT=args.anchor_assignment
    (OUT/'inputs.json').write_text(json.dumps({'run':str(RUN),'report':str(REPORT) if REPORT else None,'anchor_assignment':ANCHOR_ASSIGNMENT},indent=2))
    inventory()
    if args.replay:replay()
