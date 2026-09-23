"""Exercise the real split command on synthetic patient-level rows."""
import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec('sklearn'), 'requires scikit-learn')
class CVTests(unittest.TestCase):
    def test_heldout_is_preserved_and_pool_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / 'source.csv'
            rows = [{'case_id': f'synthetic_{i}', 'split': 'train',
                     'label': i % 2, 'birads': '4A'} for i in range(12)]
            rows.append({'case_id': 'heldout', 'split': 'test', 'label': 1, 'birads': '5'})
            with source.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['case_id', 'split', 'label', 'birads'])
                writer.writeheader()
                writer.writerows(rows)
            command = [sys.executable, str(ROOT / 'create_multitask_cv_manifests.py'),
                       '--source-manifest', str(source), '--output-dir', str(folder / 'folds')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            validation_ids = set()
            for path in sorted((folder / 'folds').glob('*.csv')):
                with path.open(newline='', encoding='utf-8') as stream:
                    fold = list(csv.DictReader(stream))
                self.assertEqual([r['split'] for r in fold if r['case_id'] == 'heldout'], ['test'])
                ids = {r['case_id'] for r in fold if r['split'] == 'val'}
                self.assertTrue(validation_ids.isdisjoint(ids))
                validation_ids.update(ids)
            self.assertEqual(len(validation_ids), 12)
            result = subprocess.run(command + ['--pool-splits', 'train', 'test'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('disjoint', result.stderr)


if __name__ == '__main__':
    unittest.main()
