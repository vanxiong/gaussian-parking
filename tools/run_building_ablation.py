"""Fresh stationary building-present/absent captures and repeated paired scoring."""
import argparse,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from run_occupancy_evaluation import ROOT,linux

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repeats',type=int,default=20);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--present-report',type=Path);p.add_argument('--absent-report',type=Path);p.add_argument('--output',type=Path);p.add_argument('--anchor-assignment',choices=['random','deterministic'],required=True);a=p.parse_args()
    if a.repeats<2:p.error('At least two paired repeats required')
    if bool(a.present_report)!=bool(a.absent_report):p.error('Supply both report directories to reuse captures')
    output=(a.output or ROOT/'reports/building_ablation'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')).resolve()
    output.mkdir(parents=True,exist_ok=False)
    plan={'experiment':'building_presence','variants':['present','absent'],'repeats_per_variant':a.repeats,'first_seed':a.seed,'warmup_per_variant':2,
        'car_appearance':'realistic','car_ids':list(range(10)),'capture_pose':'Default parking-lot start; XY (-5,-2) m; yaw 0; settled, paused physics',
        'scoring':'Unchanged above-ground ROI, raw class 17 empty; per-variant ground truth',
        'secondary_scoring':'Same ROI excluding every voxel removed from ground truth with the building; identical truth elsewhere',
        'execution_order':'Matching seed within each pair, alternating present-first and absent-first',
        'scope':'One pose; repeated inference measures assignment/runtime variation, not scene generalization',
        'anchor_assignment':a.anchor_assignment,'created_utc':datetime.now(timezone.utc).isoformat()}
    (output/'analysis-plan.json').write_text(json.dumps(plan,indent=2));reports={}
    for presence,existing in [('present',a.present_report),('absent',a.absent_report)]:
        if existing:reports[presence]=existing.resolve();continue
        print('CAPTURING building '+presence+'; all 10 cars realistic',flush=True)
        previous=(ROOT/'reports/evaluation/latest.json').read_text() if (ROOT/'reports/evaluation/latest.json').exists() else None
        with (output/(presence+'-capture-evaluation.log')).open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'tools/run_occupancy_evaluation.py'),'--stationary','--parked-car-appearance','realistic',
                '--building',presence,'--repeats','1','--anchor-assignment',a.anchor_assignment],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        current=(ROOT/'reports/evaluation/latest.json').read_text()
        if current==previous:raise RuntimeError('No new capture evaluation published')
        reports[presence]=Path(json.loads(current)['manifest']).parent
        (output/'source-reports.json').write_text(json.dumps({k:str(v) for k,v in reports.items()},indent=2))
        print('CAPTURE_READY '+presence+' '+str(reports[presence]),flush=True)
    request={**plan,'manifests':{k:linux(v/'manifest.json') for k,v in reports.items()}}
    (output/'comparison-manifest.json').write_text(json.dumps(request,indent=2))
    subprocess.run([str(ROOT/'tools/prepare_runtime.bat')],cwd=ROOT,check=True)
    print('BUILDING_COMPARISON '+str(output),flush=True)
    with (output/'replay.log').open('w') as log:
        subprocess.run(['wsl.exe','--distribution','Ubuntu-24.04','--exec','bash',linux(ROOT/'tools/compare_building_ablation.sh'),linux(output/'comparison-manifest.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/validate_car_appearance_comparison.py'),str(output/'comparison.json')],cwd=ROOT,check=True)
    (ROOT/'reports/building_ablation/latest.json').parent.mkdir(parents=True,exist_ok=True)
    (ROOT/'reports/building_ablation/latest.json').write_text(json.dumps({'report':str(output/'COMPARISON.md'),'results':str(output/'comparison.json')},indent=2))
    print((output/'COMPARISON.md').read_text(encoding='utf-8'),flush=True)

if __name__=='__main__':main()
