"""Exercise fresh-clone orchestration without a simulator, GPU or saved reports."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_multiview_ablations as multiview
import run_sensor_ablations as sensor
import run_car_appearance_ablation as appearance
import run_building_ablation as building


class FreshCloneTests(unittest.TestCase):
    def test_multiview_captures_original_and_new_views_and_records_policy(self):
        for policy in ('random', 'deterministic'):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                poses = root / 'poses.json'
                poses.write_text(json.dumps([{'id': 'west', 'x': -12, 'y': 0, 'yaw_deg': 0}]))
                output = root / 'reports/multiview_ablation/test'
                calls = []

                def execute(command, **kwargs):
                    calls.append(command)
                    if '--stationary-poses' in command:
                        views = json.loads(Path(command[command.index('--stationary-poses') + 1]).read_text())
                        variant = command[command.index('--parked-car-appearance') + 1]
                        run = root / 'simulation/data' / f'run-{len(calls)}-{variant}'
                        run.mkdir(parents=True)
                        entries = []
                        for view in views:
                            capture = run / view['id']
                            capture.mkdir()
                            entries.append({'view': view, 'capture': str(capture)})
                        (run / 'result.json').write_text(json.dumps({'status': 'stationary_evaluation', 'stationary_views': entries}))
                        (run.parent / 'latest_driving_run.json').write_text(json.dumps({'run': str(run)}))

                argv = ['run', '--poses', str(poses), '--output', str(output),
                        '--repeats', '2', '--anchor-assignment', policy]
                with patch.object(multiview, 'ROOT', root), patch.object(multiview, 'linux', str), \
                     patch.object(multiview.subprocess, 'run', side_effect=execute), \
                     patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                    multiview.main()
                manifest = json.loads((output / 'comparison-manifest.json').read_text())
                self.assertEqual(manifest['anchor_assignment'], policy)
                self.assertEqual(set(manifest['samples']), {'original', 'west'})
                for samples in manifest['samples'].values():
                    self.assertEqual(set(samples), set(multiview.CAPTURE_VARIANTS))
                self.assertEqual(sum('--stationary-poses' in call for call in calls), 4)
                # Refuse to resume the same experiment under another policy.
                argv = ['run', '--poses', str(poses), '--resume', str(output), '--repeats', '2',
                        '--anchor-assignment', 'deterministic' if policy == 'random' else 'random']
                with patch.object(multiview, 'ROOT', root), patch.object(sys, 'argv', argv):
                    with self.assertRaisesRegex(ValueError, 'Resume plan differs'):
                        multiview.main()

    def test_sensor_generates_both_captures_without_saved_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'reports/sensor_ablation/test'
            captures = []

            def execute(command, **kwargs):
                if '--stationary' in command:
                    self.assertIn('--anchor-assignment', command)
                    self.assertEqual(command[command.index('--anchor-assignment') + 1], 'random')
                    width = command[command.index('--camera-width') + 1]
                    captures.append(width)
                    report = root / 'reports/evaluation' / width
                    report.mkdir(parents=True)
                    (report / 'manifest.json').write_text('{}')
                    (report.parent / 'latest.json').write_text(json.dumps({'manifest': str(report / 'manifest.json')}))
                if command[0] == 'wsl.exe':
                    (output / 'COMPARISON.md').write_text('Fixture result')

            argv = ['run', '--output', str(output), '--repeats', '2', '--anchor-assignment', 'random']
            with patch.object(sensor, 'ROOT', root), patch.object(sensor, 'linux', str), \
                 patch.object(sensor.subprocess, 'run', side_effect=execute), \
                 patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                sensor.main()
            self.assertEqual(captures, ['640', '960'])
            manifest = json.loads((output / 'comparison-manifest.json').read_text())
            self.assertEqual(manifest['anchor_assignment'], 'random')
            self.assertEqual(set(manifest['manifests']), {'baseline', 'high_resolution'})

    def test_experiments_require_explicit_anchor_policy(self):
        for module in (multiview, sensor, appearance, building):
            with self.subTest(module=module.__name__), patch.object(sys, 'argv', ['run']), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    module.main()
                self.assertEqual(error.exception.code, 2)

    def test_paired_studies_capture_both_variants_with_selected_policy(self):
        for module in (appearance, building):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / '.cache/custom-experiment'
                captures = []

                def execute(command, **kwargs):
                    if '--stationary' in command:
                        self.assertEqual(command[command.index('--anchor-assignment') + 1], 'deterministic')
                        captures.append(command)
                        report = root / 'reports/evaluation' / str(len(captures))
                        report.mkdir(parents=True)
                        (report / 'manifest.json').write_text('{}')
                        (report.parent / 'latest.json').write_text(json.dumps({'manifest': str(report / 'manifest.json')}))
                    if command[0] == 'wsl.exe':
                        (output / 'COMPARISON.md').write_text('Fixture result')

                argv = ['run', '--output', str(output), '--repeats', '2',
                        '--anchor-assignment', 'deterministic']
                with patch.object(module, 'ROOT', root), patch.object(module, 'linux', str), \
                     patch.object(module.subprocess, 'run', side_effect=execute), \
                     patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                    module.main()
                self.assertEqual(len(captures), 2)
                manifest = json.loads((output / 'comparison-manifest.json').read_text())
                self.assertEqual(manifest['anchor_assignment'], 'deterministic')


if __name__ == '__main__':
    unittest.main()
