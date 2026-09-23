"""Launch training from a JSON configuration without machine-specific paths."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/manuscript_method.json')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', default='outputs/run')
    parser.add_argument('--pretrained-path')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    config.update(manifest_csv=str(Path(args.manifest).resolve()), output_dir=str(Path(args.output).resolve()))
    if args.pretrained_path:
        config['pretrained_path'] = str(Path(args.pretrained_path).resolve())
    if args.pretrained:
        config['pretrained'] = True
    command = [sys.executable, str(root / 'train_multitask_dual_view.py')]
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                command.append('--' + key)
        elif value is not None:
            command.append('--' + key)
            command.extend(map(str, value if isinstance(value, list) else [value]))
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()
