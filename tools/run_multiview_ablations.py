"""Capture all configured viewpoints afresh, then replay matched ablations."""
import argparse,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from run_occupancy_evaluation import ROOT,linux

CAPTURE_VARIANTS={
    'simple':{'appearance':'simple','building':'present','width':640},
    'reference':{'appearance':'realistic','building':'present','width':640},
    'no_building':{'appearance':'realistic','building':'absent','width':640},
    'high_resolution':{'appearance':'realistic','building':'present','width':960}}

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    p=argparse.ArgumentParser();p.add_argument('--poses',type=Path,default=ROOT/'configs/ablation_viewpoints.json')
    p.add_argument('--repeats',type=int,default=20);p.add_argument('--seed',type=int,default=42);p.add_argument('--output',type=Path);p.add_argument('--resume',type=Path);p.add_argument('--anchor-assignment',choices=['random','deterministic'],required=True);a=p.parse_args()
    if a.repeats<2:p.error('At least two repeats required')
    output=(a.resume or a.output or ROOT/'reports/multiview_ablation'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')).resolve()
    output.mkdir(parents=True,exist_ok=bool(a.resume));views=json.loads(a.poses.read_text())
    if not any(v['id']=='original' for v in views):
        views.insert(0,{'id':'original','x':-5.,'y':-2.,'yaw_deg':0.})
    if len({v['id'] for v in views}) != len(views):raise ValueError('Viewpoint IDs must be unique')
    capture_poses=output/'capture-poses.json'
    plan={'experiment':'multiview_ablations','repeats_per_condition':a.repeats,'first_seed':a.seed,'warmup_per_condition':2,
        'poses':views,
        'captured_variants':CAPTURE_VARIANTS,'replay_variant':'current_sweep derived from reference point input, time lag zero only',
        'comparison_direction':{'appearance':'reference - simple','building':'no_building - reference','resolution':'high_resolution - reference','history':'current_sweep - reference'},
        'scoring':'Same primary above-ground policy; building also scored outside geometry-removed cells',
        'aggregation':'Paired effects within each pose; equal-pose mean, range, sign counts across poses. Replay seeds are not independent viewpoints.',
        'original_pose_policy':'Capture original and all configured poses afresh; resume reuses only this experiment capture state',
        'anchor_assignment':a.anchor_assignment,'created_utc':datetime.now(timezone.utc).isoformat()}
    plan_path=output/'analysis-plan.json'
    if plan_path.exists():
        old=json.loads(plan_path.read_text())
        if any(old.get(k)!=plan.get(k) for k in ['poses','repeats_per_condition','first_seed','anchor_assignment','captured_variants']):raise ValueError('Resume plan differs')
    else:plan_path.write_text(json.dumps(plan,indent=2))
    capture_poses.write_text(json.dumps(views,indent=2))
    state_path=output/'capture-runs.json';runs=json.loads(state_path.read_text()) if state_path.exists() else {}
    subprocess.run([str(ROOT/'tools/prepare_runtime.bat')],cwd=ROOT,check=True)
    for variant,setting in CAPTURE_VARIANTS.items():
        if variant in runs:continue
        print('CAPTURING_BATCH',variant,len(views),'viewpoints',flush=True)
        with (output/(variant+'-capture.log')).open('w') as log:
            subprocess.run([str(ROOT/'runtime/python.bat'),str(ROOT/'simulation/driving_scene.py'),'--headless','--stationary-evaluation',
                '--stationary-poses',str(capture_poses),'--parked-car-appearance',setting['appearance'],
                '--building',setting['building'],'--camera-width',str(setting['width'])],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        run=Path(json.loads((ROOT/'simulation/data/latest_driving_run.json').read_text())['run'])
        result=json.loads((run/'result.json').read_text())
        if result['status']!='stationary_evaluation' or len(result['stationary_views'])!=len(views):raise RuntimeError('Incomplete fixed-pose batch')
        runs[variant]=str(run);state_path.write_text(json.dumps(runs,indent=2))
        print('CAPTURE_BATCH_READY',variant,str(run),flush=True)
    samples={v['id']:{} for v in plan['poses']};jobs=[]
    for variant,run_path in runs.items():
        run=Path(run_path);result=json.loads((run/'result.json').read_text())
        for entry in result['stationary_views']:
            pose_id=entry['view']['id'];capture=Path(entry['capture'])
            geometry=output/'geometry'/f'{pose_id}-{variant}.json'
            jobs.append({'scene':str(capture/'scene.usda'),'output':str(geometry)})
            samples[pose_id][variant]={'capture':linux(capture),'geometry':linux(geometry)}
    (output/'export-jobs.json').write_text(json.dumps(jobs,indent=2))
    if not all(Path(job['output']).is_file() for job in jobs):
        print('EXPORTING_GEOMETRY',len(jobs),'snapshots',flush=True)
        with (output/'geometry-export.log').open('w') as log:
            subprocess.run([str(ROOT/'runtime/python.bat'),str(ROOT/'tools/export_eval_geometry.py'),'--batch',str(output/'export-jobs.json'),'--no-ros-env'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    request={**json.loads(plan_path.read_text()),'samples':samples}
    (output/'comparison-manifest.json').write_text(json.dumps(request,indent=2))
    print('REPLAYING_ALL_VIEWPOINTS',str(output),flush=True)
    with (output/'replay.log').open('w') as log:
        subprocess.run(['wsl.exe','--distribution','Ubuntu-24.04','--exec','bash',linux(ROOT/'tools/compare_multiview_ablations.sh'),linux(output/'comparison-manifest.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/validate_multiview_ablations.py'),str(output/'comparison.json')],cwd=ROOT,check=True)
    (ROOT/'reports/multiview_ablation/latest.json').parent.mkdir(parents=True,exist_ok=True)
    (ROOT/'reports/multiview_ablation/latest.json').write_text(json.dumps({'report':str(output/'VIEWPOINT_RESULTS.md'),'results':str(output/'comparison.json')},indent=2))
    print('MULTIVIEW_READY',str(output/'VIEWPOINT_RESULTS.md'),flush=True)

if __name__=='__main__':main()
