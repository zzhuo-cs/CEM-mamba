import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('validator', ROOT / 'scripts/validate_manifest.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class ManifestTests(unittest.TestCase):
    def test_synthetic_schema(self):
        self.assertEqual(validator.validate(ROOT / 'examples/manifest.example.csv', False), [])

    def test_duplicate_and_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'manifest.csv'
            path.write_text('case_id,split,label,birads,cc_roi_path,mlo_roi_path\n'
                            'same,train,0,4A,shared.png,a.png\n'
                            'same,test,1,5,shared.png,b.png\n', encoding='utf-8')
            errors = validator.validate(path, False)
            self.assertTrue(any('duplicate' in e for e in errors))
            self.assertTrue(any('across splits' in e for e in errors))

    def test_rejects_invalid_labels_and_missing_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'manifest.csv'
            path.write_text('case_id,split,label,birads,cc_roi_path,mlo_roi_path\n'
                            'case,train,2,6,a.png,\n', encoding='utf-8')
            errors = validator.validate(path, False)
            self.assertEqual(len(errors), 3)


if __name__ == '__main__':
    unittest.main()
