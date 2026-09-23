"""Validate paired-image manifests without importing model dependencies."""
import argparse
import csv
from pathlib import Path


def validate(path, check_files=True):
    errors = []
    with open(path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'case_id', 'split', 'label', 'birads', 'cc_roi_path', 'mlo_roi_path'}
        missing = required - set(reader.fieldnames or [])
        if missing:
            return [f'Missing columns: {sorted(missing)}']
        rows = list(reader)
    if not rows:
        errors.append('Manifest is empty')
    seen = set()
    image_splits = {}
    for line, row in enumerate(rows, 2):
        case = row['case_id'].strip()
        split = row['split'].strip()
        if not case or case in seen:
            errors.append(f'Row {line}: empty or duplicate case_id')
        seen.add(case)
        if not split:
            errors.append(f'Row {line}: missing split')
        if row['label'] not in {'0', '1'}:
            errors.append(f'Row {line}: label must be 0 or 1')
        birads = row['birads'].strip().upper()
        if birads and birads not in {'3', '4A', '4B', '4C', '5'}:
            errors.append(f'Row {line}: unsupported BI-RADS value')
        if split in {'train', 'val'} and not birads:
            errors.append(f'Row {line}: development rows require BI-RADS')
        for view in ['cc', 'mlo']:
            value = row[f'{view}_roi_path'].strip()
            if not value:
                errors.append(f'Row {line}: missing {view} ROI path')
                continue
            image = Path(value).resolve()
            if check_files and not image.is_file():
                errors.append(f'Row {line}: missing image {value}')
            if image in image_splits and image_splits[image] != split:
                errors.append(f'Row {line}: image shared across splits')
            image_splits[image] = split
    return errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest')
    parser.add_argument('--skip-files', action='store_true', help='Schema checks only; for the synthetic example')
    args = parser.parse_args()
    errors = validate(args.manifest, not args.skip_files)
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Manifest checks passed.')
