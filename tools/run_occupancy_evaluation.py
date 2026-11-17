"""Windows orchestrator for reproducible simulator-geometry occupancy evaluation."""
import argparse,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def linux(path):
    path=path.resolve();return '/mnt/'+path.drive[0].lower()+path.as_posix()[2:]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,help='Driving run captured with --evaluation-snapshots; default latest paired run')
    parser.add_argument('--frames',type=int,default=5);parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--warmup',type=int,default=2)
    args=parser.parse_args()
    if args.frames<1 or args.repeats<1 or args.warmup<1:parser.error('Counts must be positive')
    if args.run is None:
        paired=[path for path in sorted((ROOT/'simulation/data/driving_runs').glob('*')) if any(path.glob('frame_*/scene.usda'))]
        if not paired:raise RuntimeError('Capture a run with --evaluation-snapshots first')
        args.run=paired[-1]
    run=args.run.resolve();run.relative_to(ROOT/'simulation/data/driving_runs')
    candidates=[p for p in sorted(run.glob('frame_*')) if (p/'scene.usda').is_file() and (p/'raw_labels.npy').is_file()]
    if len(candidates)<args.frames:raise RuntimeError('Not enough paired frames. Capture with drive_gaussianformer3d.bat --headless --evaluation-snapshots')
    selected=[candidates[i] for i in np.linspace(0,len(candidates)-1,args.frames,dtype=int)]
    output=ROOT/'reports/evaluation'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');output.mkdir(parents=True)
    jobs=[];samples=[]
    for capture in selected:
        meta=json.loads((capture/'capture.json').read_text())
        if meta.get('ground_truth_snapshot',{}).get('used_as_model_input') is not False:raise RuntimeError('Missing independent synchronized ground-truth provenance')
        geometry=output/(capture.name+'-geometry.json')
        jobs.append({'scene':str(capture/'scene.usda'),'output':str(geometry)})
        samples.append({'capture':linux(capture),'geometry':linux(geometry)})
    manifest={'run':linux(run),'samples':samples,'repeats':args.repeats,'warmup':args.warmup,
        'limitations':['One simplified synthetic parking scene; correlated poses, not an independent test set.',
                      'Uses filled rendered geometry; not the official SurroundOcc/nuScenes annotation protocol.',
                      'Full scene completion includes occluded voxels; no sensor visibility mask.',
                      'Curved primitives are tessellated; exact voxel IoU is sensitive to 0.5 m grid boundaries.',
                      'No semantic mIoU or real-world generalization claim.']}
    (output/'export-jobs.json').write_text(json.dumps(jobs,indent=2));(output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    subprocess.run([str(ROOT/'tools/prepare_runtime.bat')],check=True,cwd=ROOT)
    with (output/'geometry-export.log').open('w') as log:
        subprocess.run([str(ROOT/'runtime/python.bat'),str(ROOT/'tools/export_eval_geometry.py'),'--batch',
            str(output/'export-jobs.json'),'--no-ros-env'],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    print('EVALUATION_REPORT '+str(output),flush=True)
    with (output/'evaluation.log').open('w') as log:
        subprocess.run(['wsl.exe','--distribution','Ubuntu-24.04','--exec','bash',linux(ROOT/'tools/evaluate_occupancy.sh'),
            linux(output/'manifest.json')],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    print((output/'REPORT.md').read_text(),flush=True)
    (ROOT/'reports/evaluation/latest.json').write_text(json.dumps({'report':str(output/'REPORT.md'),'manifest':str(output/'manifest.json')},indent=2))

if __name__=='__main__':main()
