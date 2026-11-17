"""Inspect tensors with safe loading; this does not validate model compatibility."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
checkpoint = ROOT / 'checkpoints/surroundocc_release.pth'
loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
state = loaded.get('state_dict', loaded) if isinstance(loaded, dict) else loaded
if not isinstance(state, dict) or not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
    raise RuntimeError('Expected a nonempty tensor state dictionary')
if not all(torch.isfinite(value).all().item() for value in state.values() if value.is_floating_point()):
    raise RuntimeError('Nonfinite checkpoint tensors')
report = {'checkpoint': checkpoint.name,
          'sha256': hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest(),
          'tensor_loading_verified': True, 'state_dict_tensors': len(state),
          'parameters_and_buffers': sum(value.numel() for value in state.values()),
          'example_keys': list(state)[:12], 'strict_model_matching_verified': False,
          'cuda_available': torch.cuda.is_available()}
if report['cuda_available']:
    x = torch.tensor([2.0], device='cuda')
    assert (x * x).item() == 4.0
    report['cuda_tensor_execution_verified'] = True
    report['gpu'] = torch.cuda.get_device_name()
(ROOT / 'reports/checkpoint_verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
