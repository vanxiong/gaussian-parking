"""Summarize saved driving runs without copying large trajectories into reports."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = []
for path in sorted((ROOT / 'simulation/data/driving_runs').glob('*/result.json')):
    result = json.loads(path.read_text())
    rows.append({
        'run': path.parent.name,
        **{key: result.get(key) for key in (
            'scenario', 'status', 'position_error_m', 'heading_error_deg',
            'final_speed_mps', 'simulation_seconds', 'collision_audit_obstacle')},
        'stale_brake_steps': result.get('async_metrics', {}).get('stale_brake_steps'),
        'moving_steps_while_pending': result.get('async_metrics', {}).get('moving_steps_while_pending'),
    })
(ROOT / 'reports/driving_validation.json').write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
