"""Read-only GaussianFormer3D readiness check; no simulator or inference startup."""
import argparse
import importlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/environment.json')
    args = parser.parse_args()
    report = {'python': sys.version, 'executable': sys.executable,
              'platform': platform.platform(), 'packages': {},
              'nvcc': shutil.which('nvcc'), 'checkpoint_exists':
              (ROOT / 'checkpoints/surroundocc_release.pth').is_file()}
    for name in ['torch', 'mmengine', 'mmcv', 'mmdet', 'spconv']:
        try:
            module = importlib.import_module(name)
            report['packages'][name] = {'version': getattr(module, '__version__', 'unknown')}
            if name == 'torch':
                report['cuda_available'] = module.cuda.is_available()
                if report['cuda_available']:
                    report['gpu'] = module.cuda.get_device_name()
                    report['gpu_capability'] = module.cuda.get_device_capability()
                    report['torch_cuda_arches'] = module.cuda.get_arch_list()
        except Exception as error:
            report['packages'][name] = {'error': str(error)}
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                 '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
        report['nvidia_smi'] = result.stdout.strip() or result.stderr.strip()
    except Exception as error:
        report['nvidia_smi'] = str(error)
    result = subprocess.run(['git', '-C', str(ROOT / 'GaussianFormer3D'), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True)
    report['upstream_revision'] = result.stdout.strip()
    report['dependency_imports_ok'] = all('error' not in item for item in report['packages'].values())
    def read_report(name):
        path=ROOT/'reports'/name
        return json.loads(path.read_text()) if path.is_file() else {}
    build=read_report('model_build.json')
    smoke=read_report('cuda_smoke.json')
    report['strict_frozen_model_verified'] = build.get('strict_tensor_count')==941 and build.get('frozen') is True
    checks=['modulated_deform_conv_zero_offset_parity','voxelization_counts','hard_voxel_feature_mean',
            'spconv_gpu_forward','gaussian_aggregate_gpu_forward','sparse_lidar_anchor_assignment','dfa3d_symbols_present']
    report['cuda_operators_verified'] = all(smoke.get(name) is True for name in checks)
    latest=read_report('latest_prediction.json')
    prediction=ROOT/latest['prediction']/'prediction.json' if latest.get('prediction') else None
    previous=json.loads(prediction.read_text()) if prediction and prediction.is_file() else {}
    report['last_sim_inference_input_validated'] = previous.get('input_validation_passed') is True
    report['ready_for_inference'] = bool(report['dependency_imports_ok'] and report.get('cuda_available')
        and report['checkpoint_exists'] and report['strict_frozen_model_verified'] and report['cuda_operators_verified'])
    report['remaining_checks'] = []
    if not report['cuda_operators_verified']:
        report['remaining_checks'].append('GPU operator verification')
    if not report['strict_frozen_model_verified']:
        report['remaining_checks'].append('Strict frozen checkpoint loading')
    if not report['last_sim_inference_input_validated']:
        report['remaining_checks'].append('Inference on validated simulator camera/LiDAR inputs')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
