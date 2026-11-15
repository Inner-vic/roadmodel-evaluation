"""Read-only mount planning from the submitted dataset link, using official CLI.

No space start/stop, raw data downloads, PVC synthesis, or default asset version.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcm_eval.dataset_mount import resolve_dataset, mount_plan
from pcm_eval.lpai import request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-ref', required=True)
    parser.add_argument('--space-name', required=True)
    parser.add_argument('--cli', default='lpai-agent')
    args = parser.parse_args()
    query = lambda a: request(a, cli=args.cli)
    asset = resolve_dataset(args.dataset_ref, query)
    space = query(['devspaces', 'get', '--namespace', asset['namespace'], '--name', args.space_name])
    print(json.dumps(mount_plan(asset, space), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
