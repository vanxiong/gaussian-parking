"""Summarize matched startup benchmarks and independently validated simulator reuse."""
import argparse,json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('first',type=Path);p.add_argument('second',type=Path);args=p.parse_args()
runs=[json.loads(path.read_text()) for path in [args.first,args.second]]
assert all(run['passed'] and run['frozen_model_verified'] for run in runs)
services=[run['model_service'] for run in runs]
assert services[0]['pid']==services[1]['pid']
assert services[1]['session']>services[0]['session']
assert all(service['model_load_count']==1 for service in services)
assert not services[0]['reused_at_start'] and services[1]['reused_at_start']
benchmarks={mode:[json.loads((root/f'reports/performance/{mode}-{i}.json').read_text()) for i in [1,2]] for mode in ['original','fast']}
for mode,items in benchmarks.items():
    assert all(item['strict_tensor_count']==941 and item['frozen'] for item in items)
    assert all(('.runtime-linux/mount/' in item['torch_path'])==(mode=='fast') for item in items)
medians={mode:statistics.median(item['seconds'] for item in items) for mode,items in benchmarks.items()}
summary={'method':'Two alternating fresh-process runs per environment; warm OS caches, no cache flush; simulators closed.',
    'startup_seconds':{mode:[item['seconds'] for item in items] for mode,items in benchmarks.items()},
    'median_startup_seconds':medians,'startup_speedup':medians['original']/medians['fast'],
    'startup_reduction_percent':100*(1-medians['fast']/medians['original']),
    'same_model_pid':services[0]['pid'],'model_load_count':1,
    'sessions':[s['session'] for s in services],
    'session_first_prediction_seconds':[s['session_first_prediction_seconds'] for s in services],
    'whole_run_wall_seconds':[run['supervisor_wall_seconds'] for run in runs],
    'parking_error_m':[run['position_error_m'] for run in runs],
    'validation_reports':[str(args.first),str(args.second)],'passed':True}
target=root/'reports/performance/startup-summary.json';target.write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
