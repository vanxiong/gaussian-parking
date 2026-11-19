"""Windows orchestrator for reproducible simulator-geometry occupancy evaluation."""
import argparse,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def linux(path):
    path=path.resolve();return '/mnt/'+path.drive[0].lower()+path.as_posix()[2:]

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,help='Driving run captured with --evaluation-snapshots; default latest paired run')
    parser.add_argument('--stationary',action='store_true',help='Capture a fresh fixed-pose parking sample, then evaluate it')
    parser.add_argument('--parked-car-appearance',choices=['simple','realistic'],default='simple',help='Stationary capture appearance variant')
    parser.add_argument('--building',choices=['present','absent'],default='present',help='Stationary parking-lot building presence')
    parser.add_argument('--camera-width',type=int,choices=[480,640,960],default=640,help='Native camera render width for a fresh stationary capture')
    parser.add_argument('--start-offset',type=float,default=0.)
    parser.add_argument('--start-y-offset',type=float,default=0.)
    parser.add_argument('--start-yaw-deg',type=float,default=0.)
    parser.add_argument('--frames',type=int,default=5);parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--anchor-assignment',choices=['random','deterministic'],default='deterministic')
    args=parser.parse_args()
    if not args.stationary and args.parked_car_appearance!='simple':parser.error('--parked-car-appearance selects a fresh --stationary capture; replay reads the recorded variant from --run')
    if not args.stationary and args.building!='present':parser.error('--building selects a fresh --stationary capture; replay reads recorded presence from --run')
    if not args.stationary and args.camera_width!=640:parser.error('--camera-width selects a fresh --stationary capture; replay uses saved images')
    if args.frames<1 or args.repeats<1 or args.warmup<1:parser.error('Counts must be positive')
    if args.stationary:
        if args.run is not None:parser.error('--stationary captures a fresh run; omit --run')
        subprocess.run([str(ROOT/'tools/prepare_runtime.bat')],check=True,cwd=ROOT)
        capture_log=ROOT/'reports/stationary-capture.log';capture_log.parent.mkdir(parents=True,exist_ok=True)
        with capture_log.open('w') as log:
            subprocess.run([str(ROOT/'runtime/python.bat'),str(ROOT/'simulation/driving_scene.py'),
                '--headless','--stationary-evaluation','--screenshot','--parked-car-appearance',args.parked_car_appearance,'--building',args.building,'--camera-width',str(args.camera_width),'--start-offset',str(args.start_offset),
                '--start-y-offset',str(args.start_y_offset),'--start-yaw-deg',str(args.start_yaw_deg)],
                stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        args.run=Path(json.loads((ROOT/'simulation/data/latest_driving_run.json').read_text())['run'])
        result=json.loads((args.run/'result.json').read_text())
        if result['status']!='stationary_evaluation':raise RuntimeError('Capture was not a stationary evaluation')
        args.frames=1
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
    manifest={'anchor_assignment':args.anchor_assignment,'run':linux(run),'samples':samples,'repeats':args.repeats,'warmup':args.warmup,
        'limitations':['One simplified synthetic parking scene; correlated poses, not an independent test set.',
                      'Uses filled rendered geometry; not the official SurroundOcc/nuScenes annotation protocol.',
                      'Full scene completion includes occluded voxels; no sensor visibility mask.',
                      'Curved primitives are tessellated; exact voxel IoU is sensitive to 0.5 m grid boundaries.',
                      'No semantic mIoU or real-world generalization claim.']}
    result=json.loads((run/'result.json').read_text())
    manifest['appearance_ablation']=result.get('appearance_ablation',{'variant':'simple','ground_truth_policy':'rendered_geometry'})
    manifest['building_ablation']=result.get('building_ablation',{'presence':'present','recorded_in_legacy_capture':False})
    if manifest['building_ablation']['presence']=='absent':manifest['limitations'].append('Building subtree removed from rendered scene, collision/LiDAR and reference occupancy; occupied target differs from building-present runs.')
    if manifest['appearance_ablation']['variant']=='realistic':
        manifest['limitations'].append('Appearance-only ablation: selected detailed visual cars use unchanged simple-car occupancy proxies and LiDAR colliders; not exact detailed-mesh ground truth.')
    manifest['stationary_evaluation']=result.get('stationary_evaluation',False)
    if manifest['stationary_evaluation']:
        manifest['fixed_world_from_ego']=result['fixed_world_from_ego']
        manifest['limitations'][0]='One fixed pose in a simplified synthetic parking scene; not a generalization test.'
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
    (ROOT/'reports/evaluation/latest.json').parent.mkdir(parents=True,exist_ok=True)
    (ROOT/'reports/evaluation/latest.json').write_text(json.dumps({'report':str(output/'REPORT.md'),'manifest':str(output/'manifest.json')},indent=2))

if __name__=='__main__':main()
