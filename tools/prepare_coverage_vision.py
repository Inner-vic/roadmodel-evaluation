"""Materialize only four frozen image/GT pairs and launch native vision evaluation."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import load_json, save_json
from head_data_asset import client_for_reference, read_object
from model_space_transport import run, put

OUT = Path('evidence/coverage-supplement-20261029')
BASE = '/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027/coverage-20261029'


def main():
    selected = load_json(OUT / 'vision-selection.json')
    assert len(selected['selected']) == 4 and not selected['uncovered']
    client, storage = client_for_reference()
    inputs = OUT / 'vision-input'
    files = []
    new_bytes = 0
    old = Path('evidence/model-evaluation/remaining-heads-20261027/input')
    for row in selected['selected']:
        for role in ('label_path', 'image_path'):
            rel = row[role]
            path = inputs / rel
            if not path.exists():
                if (old / rel).is_file():
                    body = (old / rel).read_bytes()
                    reused = True
                else:
                    body, _ = read_object(client, storage, rel, limit=8 * 1024 * 1024)
                    new_bytes += len(body)
                    reused = False
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
            else:
                body = path.read_bytes()
                reused = True
            sha = hashlib.sha256(body).hexdigest()
            if role == 'label_path':
                assert sha == row['label_sha256']
            files.append({'path': rel, 'sha256': sha, 'bytes': len(body), 'reused': reused})
    assert len(files) == 8 and new_bytes < 32 * 1024 * 1024
    subset = OUT / 'vision-subset.json'
    files.append({'path': subset.name, 'sha256': hashlib.sha256(subset.read_bytes()).hexdigest(), 'bytes': subset.stat().st_size})
    save_json(OUT / 'vision-input-manifest.json', files)
    archive = OUT / 'vision-input.tar'
    if not archive.exists():
        with tarfile.open(archive, 'w') as tar:
            for row in files:
                path = subset if row['path'] == subset.name else inputs / row['path']
                tar.add(path, arcname=row['path'])
    put(archive, BASE + '/input.tar')
    put(OUT / 'vision-input-manifest.json', BASE + '/input-manifest.json')
    code = """import pathlib,tarfile,json,hashlib
b=pathlib.Path(%r);manifest=json.loads((b/'input-manifest.json').read_text())
with tarfile.open(b/'input.tar') as t:
 for m in t:
  assert m.isfile();p=b/'input'/m.name;assert p.resolve().is_relative_to((b/'input').resolve());raw=t.extractfile(m).read();p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists() or p.read_bytes()==raw
  if not p.exists():p.write_bytes(raw)
assert all(hashlib.sha256((b/'input'/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in manifest)
print(json.dumps({'state':'verified','files':len(manifest)}))
""" % BASE
    print(run(['python3', '-c', code]).decode(), flush=True)
    launch = load_json('evidence/model-evaluation/remaining-heads-20261027/vision-launch.json')
    launch.update(run_id='model-v3.2.2-vision-coverage-20261029', expected_frames=4,
                  selected_frames=[[r['scene'], r['frame']['frame_index']] for r in selected['selected']])
    launch['environment']['DATA_LINK_FILE'] = BASE + '/input/vision-subset.json'
    launch['environment']['DATASET_PATH'] = BASE + '/input/DataReorganized-VisionHead,' + BASE + '/input/DataReorganized-VisionHead-standard'
    launch['argv'][launch['argv'].index('--work-dir') + 1] = BASE + '/vision-01/output'
    save_json(OUT / 'vision-launch.json', launch)
    put(OUT / 'vision-launch.json', BASE + '/vision-01/launch.json')
    put('tools/head_preflight_remote.py', BASE + '/head_preflight.py')
    worker = Path('tools/run_coverage_vision_remote.py')
    put(worker, BASE + '/run_vision.py')
    code = "import pathlib,subprocess,sys,json;b=pathlib.Path(%r);assert not (b/'vision-01/execution_receipt.json').exists();log=(b/'workflow-console.log').open('xb');p=subprocess.Popen([sys.executable,str(b/'run_vision.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({'pid':p.pid,'directory':str(b)}))" % BASE
    dispatch = json.loads(run(['python3', '-c', code]).decode())
    save_json(OUT / 'vision-dispatch.json', {'dispatch': dispatch, 'new_input_bytes': new_bytes, 'expected_frames': 4})
    print(json.dumps({'dispatch': dispatch, 'new_input_bytes': new_bytes, 'expected_frames': 4}), flush=True)


if __name__ == '__main__':
    main()
