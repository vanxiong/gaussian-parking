"""Capture both stationary appearances, then run paired repeated occupancy scoring."""
import argparse,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from run_occupancy_evaluation import ROOT,linux

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--simple-report',type=Path)
    p.add_argument('--realistic-report',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--anchor-assignment',choices=['random','deterministic'],required=True);a=p.parse_args()
    if a.repeats<2:p.error('At least two repeats are required to measure variation')
    if bool(a.simple_report)!=bool(a.realistic_report):p.error('Supply both report directories to reuse captures')
    output=a.output or ROOT/'reports/appearance_ablation'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    plan={'repeats_per_variant':a.repeats,'first_seed':a.seed,'warmup_per_variant':2,
          'scoring':'Fixed obstacles_above_ground/argmax, raw class 17 empty; occupied IoU, precision, recall',
          'capture_pose':'Default parking-lot start, offsets 0,0; yaw 0 degrees; physics paused after settling',
          'comparison':'All ten parked cars, realistic visual meshes vs simple baseline; common original geometry and LiDAR colliders',
          'execution_order':'Paired same seed per repeat, alternating simple-first and realistic-first',
          'scope':'One pose; repeated inference is assignment/runtime variation, not independent scene generalization',
          'anchor_assignment':a.anchor_assignment,'created_utc':datetime.now(timezone.utc).isoformat()}
    (output/'analysis-plan.json').write_text(json.dumps(plan,indent=2))
    reports={}
    for variant,existing in [('simple',a.simple_report),('realistic',a.realistic_report)]:
        if existing:reports[variant]=existing.resolve();continue
        print('CAPTURING '+variant+' at default fixed pose',flush=True)
        previous=(ROOT/'reports/evaluation/latest.json').read_text() if (ROOT/'reports/evaluation/latest.json').exists() else None
        with (output/(variant+'-capture-evaluation.log')).open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'tools/run_occupancy_evaluation.py'),'--stationary',
                '--parked-car-appearance',variant,'--repeats','1','--anchor-assignment',a.anchor_assignment],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
        current=(ROOT/'reports/evaluation/latest.json').read_text()
        if current==previous:raise RuntimeError('Capture did not publish a new evaluation report')
        reports[variant]=Path(json.loads(current)['manifest']).parent
        (output/'source-reports.json').write_text(json.dumps({k:str(v) for k,v in reports.items()},indent=2))
        print('CAPTURE_READY '+variant+' '+str(reports[variant]),flush=True)
    request={**plan,'manifests':{k:linux(v/'manifest.json') for k,v in reports.items()}}
    (output/'comparison-manifest.json').write_text(json.dumps(request,indent=2))
    print('REPEATED_COMPARISON '+str(output),flush=True)
    subprocess.run([str(ROOT/'tools/prepare_runtime.bat')],cwd=ROOT,check=True)
    with (output/'replay.log').open('w') as log:
        subprocess.run(['wsl.exe','--distribution','Ubuntu-24.04','--exec','bash',
            linux(ROOT/'tools/compare_car_appearance.sh'),linux(output/'comparison-manifest.json')],
            stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/validate_car_appearance_comparison.py'),str(output/'comparison.json')],cwd=ROOT,check=True)
    print((output/'COMPARISON.md').read_text(encoding='utf-8'),flush=True)
    (ROOT/'reports/appearance_ablation/latest.json').parent.mkdir(parents=True,exist_ok=True)
    (ROOT/'reports/appearance_ablation/latest.json').write_text(json.dumps({'report':str(output/'COMPARISON.md'),'results':str(output/'comparison.json')},indent=2))

if __name__=='__main__':main()
