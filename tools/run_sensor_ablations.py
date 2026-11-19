"""Capture native high-resolution input, then compare resolution and LiDAR history."""
import argparse,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from run_occupancy_evaluation import ROOT,linux

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    p=argparse.ArgumentParser();p.add_argument('--baseline-report',type=Path)
    p.add_argument('--high-resolution-report',type=Path);p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--output',type=Path);p.add_argument('--anchor-assignment',choices=['random','deterministic'],required=True);a=p.parse_args()
    if a.repeats<2:p.error('At least two repeats required')
    baseline=a.baseline_report.resolve() if a.baseline_report else None
    if baseline and not (baseline/'manifest.json').is_file():p.error('Baseline must be an existing stationary evaluation directory')
    output=(a.output or ROOT/'reports/sensor_ablation'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')).resolve();output.mkdir(parents=True,exist_ok=False)
    plan={'experiment':'sensor_inputs','variants':['baseline','high_resolution','current_sweep'],
        'repeats_per_variant':a.repeats,'first_seed':a.seed,'warmup_per_variant':2,
        'car_appearance':'all ten realistic','building':'present','pose':'XY (-5,-2) m, yaw 0, settled paused physics',
        'baseline':'640x360 native RGB, 11 LiDAR sweeps',
        'high_resolution':'960x540 native RGB capture, 11 identical LiDAR sweeps; no image upsampling',
        'current_sweep':'Identical 640x360 capture; retain only time_lag_seconds=0 input points; depth guidance already uses current sweep and is unchanged',
        'execution_order':'All six permutations cycled; matching seed reset per variant within each repeat',
        'scoring':'Fixed above-ground raw-argmax occupied IoU, precision and recall; identical ground truth',
        'scope':'One pose; repeated inference measures assignment/runtime variation, not independent scenes',
        'anchor_assignment':a.anchor_assignment,'created_utc':datetime.now(timezone.utc).isoformat()}
    (output/'analysis-plan.json').write_text(json.dumps(plan,indent=2))
    if baseline is None:
        with (output/'baseline-capture-evaluation.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'tools/run_occupancy_evaluation.py'),'--stationary',
                '--parked-car-appearance','realistic','--building','present','--camera-width','640',
                '--repeats','1','--anchor-assignment',a.anchor_assignment],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        baseline=Path(json.loads((ROOT/'reports/evaluation/latest.json').read_text())['manifest']).parent
    high=a.high_resolution_report.resolve() if a.high_resolution_report else None
    if high is None:
        print('CAPTURING native 960x540; all ten realistic cars; building present',flush=True)
        previous=(ROOT/'reports/evaluation/latest.json').read_text() if (ROOT/'reports/evaluation/latest.json').exists() else None
        with (output/'high-resolution-capture-evaluation.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'tools/run_occupancy_evaluation.py'),'--stationary','--parked-car-appearance','realistic',
                '--building','present','--camera-width','960','--repeats','1','--anchor-assignment',a.anchor_assignment],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        latest=(ROOT/'reports/evaluation/latest.json').read_text()
        if latest==previous:raise RuntimeError('No new high-resolution report')
        high=Path(json.loads(latest)['manifest']).parent
    print('CAPTURE_READY '+str(high),flush=True)
    request={**plan,'manifests':{'baseline':linux(baseline/'manifest.json'),'high_resolution':linux(high/'manifest.json')}}
    (output/'comparison-manifest.json').write_text(json.dumps(request,indent=2))
    subprocess.run([str(ROOT/'tools/prepare_runtime.bat')],cwd=ROOT,check=True)
    with (output/'replay.log').open('w') as log:
        subprocess.run(['wsl.exe','--distribution','Ubuntu-24.04','--exec','bash',linux(ROOT/'tools/compare_sensor_ablations.sh'),linux(output/'comparison-manifest.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/validate_car_appearance_comparison.py'),str(output/'comparison.json')],cwd=ROOT,check=True)
    (ROOT/'reports/sensor_ablation/latest.json').parent.mkdir(parents=True,exist_ok=True)
    (ROOT/'reports/sensor_ablation/latest.json').write_text(json.dumps({'report':str(output/'COMPARISON.md'),'results':str(output/'comparison.json')},indent=2))
    print((output/'COMPARISON.md').read_text(encoding='utf-8'),flush=True)

if __name__=='__main__':main()
