from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath

from .common import save_json, utc_now
from .parsers import collect_compile, collect_model
from .planning import ssh_command, verify_plan

REMOTE_INVENTORY = r'''
import json,pathlib,sys,hashlib
root=pathlib.Path(sys.argv[1]).resolve()
patterns=['*.log','logs/*.log','eval_scheme_report.json','vis/eval_scheme_report.json','front_boundary_metrics.json'] if sys.argv[2]=='model_accuracy' else ['metrics_board.json','metrics_onnx.json','cosine_summary.json','state.json','input_validation.json','output_integrity.json','execution_receipt.json','tensor_summary_receipt.json','effective-data-link.json','request.json','workflow.json','frame_*/frame_manifest.json','frame_*/cosine_board_onnx.json']
out=[]
for pattern in patterns:
 for p in sorted(root.glob(pattern)):
  if not p.is_file(): continue
  p.resolve().relative_to(root)
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''): h.update(b)
  out.append({'path':p.relative_to(root).as_posix(),'size':p.stat().st_size,'sha256':h.hexdigest()})
print(json.dumps(out))
'''
REMOTE_READ = r'''
import pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve(); p=(root/sys.argv[2]).resolve(); p.relative_to(root)
if p.stat().st_size>int(sys.argv[3]): raise ValueError('Result file exceeds collection limit')
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''): sys.stdout.buffer.write(b)
'''
PREFLIGHT = r'''
import os,sys,json,hashlib,subprocess
missing=[p for p in sys.argv[4:] if not os.path.exists(p)]
exists=os.path.exists(sys.argv[2])
if exists and sys.argv[1]!='resume': raise ValueError('Output directory exists; use a new run ID or explicit batch resume')
if not exists and sys.argv[1]=='resume': raise ValueError('No existing workflow to resume')
if missing: raise ValueError('Missing input paths: '+repr(missing))
identity=json.loads(sys.argv[3])
for path,expected in identity.get('files',{}).items():
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 if h.hexdigest()!=expected:raise ValueError('Input content changed: '+path)
scope=identity.get('model_scope')
if scope:
 link=json.load(open(scope['data_link_file']))
 for split in ('eval','joint_eval'):
  if not isinstance(link.get(split),dict):raise ValueError('Missing frozen evaluation split')
  actual=[[scene,frame['frame_index']] for scene,row in link[split].items() for frame in row['frames']]
  if actual!=scope['selected_frames']:raise ValueError('Datalink expands beyond or differs from frozen small scope')
repo=identity.get('repository')
if repo:
 head=subprocess.run(['git','-C',repo['path'],'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
 if head.lower()!=repo['commit_id'].lower():raise ValueError('Repository commit differs from model binding')
 if repo.get('require_clean') and subprocess.run(['git','-C',repo['path'],'diff','--quiet','HEAD','--'],check=False).returncode:
  raise ValueError('Tracked code/config changes differ from the pinned model source')
snapshot=identity.get('source_snapshot')
if snapshot:
 import pathlib
 manifest=json.loads(pathlib.Path(snapshot['manifest']).read_text())
 if manifest.get('commit')!=snapshot['commit_id']:raise ValueError('Source snapshot commit differs')
 root=pathlib.Path(snapshot['root']).resolve()
 for relative,expected in manifest['files'].items():
  path=(root/relative).resolve()
  path.relative_to(root)
  h=hashlib.sha256(path.read_bytes()).hexdigest()
  if h!=expected:raise ValueError('Pinned source file differs: '+relative)
'''


def collect_remote(alias: str, remote_run_dir: str, kind: str, destination: str | Path, *,
                   version_name: str | None = None, python: str = "python3", max_file_mb: int = 64,
                   ssh_options=None, expected_frames=None) -> dict:
    if kind not in ("model_accuracy", "compile_alignment"):
        raise ValueError("Unknown evaluation kind")
    if not PurePosixPath(remote_run_dir).is_absolute() or not 1 <= max_file_mb <= 1024:
        raise ValueError("Use an absolute remote result path and a collection limit in 1..1024 MiB")
    root = Path(destination).resolve()
    if root.exists():
        raise ValueError("Collection destination exists; choose a new directory")
    proc = subprocess.run(ssh_command(alias, [python, "-c", REMOTE_INVENTORY, remote_run_dir, kind], options=ssh_options),
                          capture_output=True, timeout=90, shell=False)
    if proc.returncode:
        raise ValueError(f"Remote result inventory failed (exit {proc.returncode})")
    inventory = json.loads(proc.stdout)
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Remote run contains no recognized result files")
    root.mkdir(parents=True)
    for item in inventory:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
            raise ValueError("Unsafe remote result path")
        if item["size"] > max_file_mb * 1024 * 1024:
            raise ValueError(f"{relative} exceeds {max_file_mb} MiB; collect explicitly with a larger limit")
        proc = subprocess.run(ssh_command(alias, [python, "-c", REMOTE_READ, remote_run_dir,
                                                  item["path"], str(max_file_mb * 1024 * 1024)], options=ssh_options),
                              capture_output=True, timeout=90, shell=False)
        if proc.returncode or len(proc.stdout) != item["size"] or hashlib.sha256(proc.stdout).hexdigest() != item["sha256"]:
            raise ValueError(f"Remote result changed or could not be collected: {relative}")
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(proc.stdout)
    save_json(root / "collection_receipt.json", {"ssh_alias": alias, "remote_run_dir": remote_run_dir,
                                               "files": inventory, "collected_at": utc_now()})
    result = (collect_model if kind == "model_accuracy" else collect_compile)(root, version_name, expected_frames=expected_frames)
    result["provenance"]["remote"] = {"ssh_alias": alias, "run_dir": remote_run_dir}
    save_json(root / "result.json", result)
    return result


def execute(plan: dict, evidence_dir: str | Path, *, resume: bool = False, batch_limit: int | None = None) -> dict:
    verify_plan(plan)
    steps = [{**step, 'argv': list(step['argv'])} for step in plan['steps']]
    if resume or batch_limit is not None:
        if (plan['kind'] != 'compile_alignment' or len(steps) != 1
                or PurePosixPath(steps[0]['argv'][1]).name != 'run_compile_batches.py'):
            raise ValueError('Resume/batch-limit are only supported by the frozen batch workflow')
        if batch_limit is not None and (type(batch_limit) is not int or batch_limit < 1):
            raise ValueError('batch-limit must be positive')
        if resume:
            steps[0]['argv'].append('--resume')
        if batch_limit is not None:
            steps[0]['argv'] += ['--batch-limit', str(batch_limit)]
    root = Path(evidence_dir).resolve()
    if root.exists():
        raise ValueError("Execution evidence directory exists; no automatic duplicate submission")
    root.mkdir(parents=True)
    save_json(root / "plan.json", plan)
    receipt = {"schema": "pcm.execution-receipt/v1", "source": plan["transport"],
               "state": "running", "started_at": utc_now(), "run_id": plan["run_id"],
               "plan_sha256": plan["plan_sha256"], "resume": resume, "batch_limit": batch_limit, "steps": []}
    save_json(root / "execution_receipt.json", receipt)
    first = steps[0]
    path_type = Path if plan["transport"] == "local" else PurePosixPath
    scripts = [str(path_type(s["cwd"]) / s["argv"][1]) for s in plan["steps"]]
    preflight = {"name": "preflight", "cwd": first["cwd"], "env": {},
                 "argv": [first["argv"][0], "-c", PREFLIGHT, 'resume' if resume else 'new',
                          plan["output_dir"], json.dumps(plan.get('input_identity', {})), *plan["input_paths"], *scripts]}
    try:
        for step in [preflight, *steps]:
            argv = (ssh_command(plan["ssh_alias"], step["argv"], step["cwd"], step["env"], options=plan.get('ssh_options'))
                    if plan["transport"] == "ssh" else step["argv"])
            start = utc_now()
            with (root / (step["name"] + ".log")).open("xb") as log:
                proc = subprocess.run(argv, cwd=step["cwd"] if plan["transport"] == "local" else None,
                                      env={**os.environ, **step["env"]} if plan["transport"] == "local" else None,
                                      stdout=log, stderr=subprocess.STDOUT, shell=False)
            receipt["steps"].append({"name": step["name"], "started_at": start,
                                      "finished_at": utc_now(), "exit_code": proc.returncode})
            if proc.returncode:
                receipt.update(state="checkpointed" if proc.returncode == 75 and batch_limit else "failed",
                               exit_code=proc.returncode)
                break
        else:
            receipt.update(state="succeeded", exit_code=0)
    except (OSError, subprocess.SubprocessError) as error:
        receipt.update(state="failed", error=str(error))
    finally:
        receipt["finished_at"] = utc_now()
        save_json(root / "execution_receipt.json", receipt, overwrite=True)
    if receipt["state"] == "succeeded":
        try:
            if plan["transport"] == "ssh":
                result = collect_remote(plan["ssh_alias"], plan["effective_output_dir"], plan["kind"],
                                        root / "collected", version_name=plan["version_name"], python=first["argv"][0],
                                        ssh_options=plan.get('ssh_options'), expected_frames=plan.get('scope',{}).get('expected_frames'))
            else:
                result = (collect_model if plan["kind"] == "model_accuracy" else collect_compile)(
                    plan["effective_output_dir"], plan["version_name"])
            result["execution"] = {"state": receipt["state"], "exit_code": receipt["exit_code"],
                                   "run_id": plan["run_id"], "source": plan["transport"]}
            result["provenance"]["execution_plan"] = plan
            save_json(root / "result.json", result)
            receipt["result_file"] = str(root / "result.json")
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            receipt["collection_error"] = str(error)
        save_json(root / "execution_receipt.json", receipt, overwrite=True)
    return receipt
