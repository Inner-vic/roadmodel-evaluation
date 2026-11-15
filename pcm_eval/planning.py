from __future__ import annotations

import math
import json
import os
import re
import shlex
import shutil
from pathlib import Path, PurePosixPath

from .common import json_digest, utc_now

KINDS = ("model_accuracy", "compile_alignment")
SAFE_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
ENV_KEYS = {"DATASET_PATH", "DATA_LINK_FILE", "DATASET", "HEAD_MODE", "TRAIN_MODE",
            "USE_TEMPORAL", "SEQUENTIAL_TRAIN", "BATCH_SIZE", "NUM_WORKERS", "ODO_SOURCE",
            "IMG_NUM_CAMS", "NUM_FRAMES", "FRAME_STRIDE", "DATA_SRC"}


def build_plan(request: dict, run_id: str | None = None) -> dict:
    if request.get('kind')=='compile_alignment' and request.get('execution',{}).get('adapter')=='subnet_chain_workstation':
        return build_subnet_plan(request,run_id)
    if request.get("kind") not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    execution = request.get("execution", {})
    dataset = request.get("dataset", {})
    model = request.get("model", {})
    evaluation = request.get("evaluation", {})
    fraction = evaluation.get("frame_fraction", 1.0)
    if isinstance(fraction, bool) or not isinstance(fraction, (float, int)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("evaluation.frame_fraction must be in (0, 1]")
    run_id = run_id or "eval-" + json_digest(request)[:16]
    if not SAFE_ALIAS.fullmatch(run_id):
        raise ValueError("run_id must contain only letters, digits, dot, underscore, hyphen")
    transport = execution.get("transport", "ssh")
    if transport not in ("local", "ssh"):
        raise ValueError("execution.transport must be local or ssh (an existing LPAI devspace is supported by SSH)")
    alias = execution.get("ssh_alias")
    if transport == "ssh" and alias is not None and not SAFE_ALIAS.fullmatch(alias):
        raise ValueError("Use a configured SSH alias, not a shell command or credentials")
    missing, issues = [], []

    def required(obj: dict, key: str, field: str):
        value = obj.get(key)
        if value is None or value == "":
            missing.append(field)
        elif not isinstance(value, str) or "\x00" in value or "\n" in value:
            raise ValueError(f"{field} must be a single nonempty string")
        return value

    version = required(request, "version_name", "version_name")
    repo = required(execution, "repo_dir", "execution.repo_dir")
    output_root = required(execution, "output_root", "execution.output_root")
    if transport == "ssh":
        required(execution, "ssh_alias", "execution.ssh_alias")
    dataset_path = required(dataset, "path", "dataset.path")
    datalink = required(dataset, "data_link_file", "dataset.data_link_file")
    dataset_class = required(dataset, "class_name", "dataset.class_name")
    path_type = PurePosixPath if transport == "ssh" else Path
    if transport == "ssh":
        for name, value in (("repo_dir", repo), ("output_root", output_root),
                            ("dataset.path", dataset_path), ("dataset.data_link_file", datalink)):
            if value and not PurePosixPath(value).is_absolute():
                raise ValueError(f"Remote {name} must be an absolute Linux path")
    output_dir = str(path_type(output_root) / run_id) if output_root else None
    python = execution.get("python", "python3" if transport == "ssh" else "python")
    if not isinstance(python, str) or not python or "\n" in python:
        raise ValueError("execution.python must be one executable path")
    env = {k: str(v) for k, v in execution.get("env", {}).items() if v is not None}
    if any(k not in ENV_KEYS for k in env):
        raise ValueError("Execution environment contains unsupported fields; use the documented model configuration keys")
    configuration_env=execution.get('configuration_env',{})
    if configuration_env:
        if not isinstance(configuration_env,dict) or not request.get('source',{}).get('require_verified_binding'):
            raise ValueError('Configuration environment requires a verified model binding')
        if request['source'].get('configuration_env_sha256')!=json_digest(configuration_env):
            raise ValueError('Configuration environment differs from its recorded binding')
        denied={'PATH','LD_PRELOAD','LD_LIBRARY_PATH','PYTHONHOME','PYTHONSTARTUP','BASH_ENV','ENV'}
        if any(not re.fullmatch('[A-Z][A-Z0-9_]*',key) or key in denied or re.search('TOKEN|PASSWORD|SECRET|CREDENTIAL|API_KEY',key)
               or not isinstance(value,str) or '\x00' in value or '\n' in value for key,value in configuration_env.items()):
            raise ValueError('Invalid or sensitive model configuration environment')
        env={**configuration_env,**env}
    env.update({k: v for k, v in {"DATASET_PATH": dataset_path, "DATA_LINK_FILE": datalink,
                                 "DATASET": dataset_class}.items() if v})
    steps, input_paths = [], [v for v in (dataset_path, datalink) if v]
    identity = {}
    effective = output_dir
    if request["kind"] == "model_accuracy":
        config = required(model, "config", "model.config")
        checkpoint = required(model, "checkpoint", "model.checkpoint")
        if checkpoint:
            input_paths.append(checkpoint)
        if config and repo:
            input_paths.append(str(path_type(repo) / config))
        if request.get('source', {}).get('require_verified_binding'):
            source = request.get('source', {})
            commit = required(source, 'commit_id', 'source.commit_id')
            if commit and not re.fullmatch('[a-fA-F0-9]{40}|[a-fA-F0-9]{64}', commit):
                raise ValueError('source.commit_id must be a full Git commit digest')
            identity = {'files': {}, 'repository': {'path': repo, 'commit_id': commit, 'require_clean': True}}
            if source.get('snapshot_manifest'):
                snapshot=required(source,'snapshot_manifest','source.snapshot_manifest')
                snapshot_sha=required(source,'snapshot_manifest_sha256','source.snapshot_manifest_sha256')
                if snapshot_sha and not re.fullmatch('[a-fA-F0-9]{64}',snapshot_sha):raise ValueError('Snapshot SHA256 required')
                identity.pop('repository')
                identity['source_snapshot']={'root':repo,'manifest':snapshot,'commit_id':commit}
                if snapshot and snapshot_sha:
                    identity['files'][snapshot]=snapshot_sha
                    input_paths.append(snapshot)
            bindings = [(model, 'checkpoint_sha256', checkpoint, 'model.checkpoint_sha256'),
                        (model, 'config_sha256', str(path_type(repo) / config) if repo and config else None, 'model.config_sha256'),
                        (dataset, 'data_link_sha256', datalink, 'dataset.data_link_sha256')]
            for key in ('trajectory_checkpoint', 'classifier_checkpoint'):
                if model.get(key):
                    bindings.append((model, key + '_sha256', model[key], 'model.' + key + '_sha256'))
            for obj, key, path, field in bindings:
                sha = required(obj, key, field)
                if sha and not re.fullmatch('[a-fA-F0-9]{64}', sha):
                    raise ValueError(field + ' must be a SHA256 digest')
                if path and sha:
                    identity['files'][path] = sha.lower()
        argv = [python, "eval.py", "--py-config", config, "--resume-from", checkpoint,
                "--work-dir", output_dir, "--eval-subsample", str(float(fraction))]
        for key, flag in (("trajectory_checkpoint", "--traj-resume"), ("classifier_checkpoint", "--classifier-resume")):
            if model.get(key):
                argv += [flag, model[key]]
                input_paths.append(model[key])
        if evaluation.get("temporal") is True:
            argv.append("--temporal-eval")
        if evaluation.get("compute_scheme", True):
            # Postprocessing needs the entire selected inference sample set.
            argv += ["--vis-occ", "--sample-freq", "1", "--zip-vis", "none"]
        if evaluation.get("scene_names"):
            names = evaluation["scene_names"]
            if not isinstance(names, list) or any(not isinstance(n, str) or "," in n for n in names):
                raise ValueError("scene_names must be an array of names without commas")
            argv += ["--scene-names", ",".join(names)]
        effective = str(path_type(output_dir) / path_type(config).stem) if output_dir and config else None
        steps.append({"name": "model_inference", "argv": argv})
        if evaluation.get("compute_scheme", True):
            steps.append({"name": "model_postprocessing", "argv": [python,
                          "tools/eval_scheme/run_eval_scheme.py", "--vis-dir",
                          str(path_type(effective) / "vis") if effective else None,
                          "--output-json", str(path_type(effective) / "eval_scheme_report.json") if effective else None]})
    else:
        onnx = required(model, "onnx", "model.onnx")
        packup = required(model, "packup_tar", "model.packup_tar")
        root = required(dataset, "export_root", "dataset.export_root")
        board = required(execution, "board_ssh", "execution.board_ssh")
        max_frames = evaluation.get("max_frames")
        if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1):
            raise ValueError("evaluation.max_frames must be a positive integer")
        if dataset_class == "SmartChassisDatasetV7" and not evaluation.get("v7_pipeline_verified", False):
            issues.append({"code": "v7_pipeline_unverified", "severity": "error",
                           "message": "The supplied benchEval NumPy pipeline is not yet validated against this V7 datalink; execution is blocked"})
        argv = [python, "run_bench_eval.py", "--eval-dataset-root", root,
                "--dataset-path", dataset_path, "--data-link-file", datalink,
                "--dataset", dataset_class, "--onnx", onnx, "--packup-tar", packup,
                "--board-ssh", board, "--save-dir", output_dir,
                "--eval-subsample", str(float(fraction) * 100),
                "--no-shm", "--transfer-archive", "--skip-git-check",
                "--board-models-root", f"/userdata/benchEval/models/{run_id}"]
        if max_frames:
            argv += ["--max-frames", str(max_frames)]
        if not evaluation.get("compute_gt_metrics", False):
            argv.append("--skip-metrics")
        if execution.get("adapter") == "workstation_driver":
            driver = required(execution, "driver_script", "execution.driver_script")
            manifest = required(dataset, "required_manifest", "dataset.required_manifest")
            for key in ("onnx_sha256", "packup_sha256"):
                value = required(model, key, "model." + key)
                if value and not re.fullmatch('[a-fA-F0-9]{64}', value):
                    raise ValueError("model." + key + " must be a SHA256 digest")
            if evaluation.get("compute_gt_metrics", False):
                issues.append({"code": "gt_driver_unsupported", "severity": "error",
                               "message": "The workstation driver is verified for tensor alignment only"})
            if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}', run_id):
                raise ValueError("The workstation driver requires a run ID of at most 64 letters/digits/underscores/hyphens")
            frozen_request = dict(request, run_id=run_id)
            argv = [python, driver, "--request-json", json.dumps(frozen_request, ensure_ascii=False)]
            if manifest:
                input_paths.append(manifest)
        input_paths += [p for p in (onnx, packup, root) if p]
        steps.append({"name": "compile_alignment", "argv": argv})
    for step in steps:
        step.update({"cwd": repo, "env": env})
    if request['kind']=='model_accuracy' and evaluation.get('selected_frames') is not None:
        selected=evaluation['selected_frames']
        if (not isinstance(selected,list) or not 1<=len(selected)<=20
                or any(not isinstance(x,list) or len(x)!=2 or not isinstance(x[0],str) or type(x[1]) is not int for x in selected)
                or len(set(map(tuple,selected)))!=len(selected)
                or evaluation.get('expected_frames')!=len(selected) or evaluation.get('max_frames')!=len(selected)):
            raise ValueError('Frozen model small scope must contain exactly the expected, unique <=20 frame identities')
        identity['model_scope']={'data_link_file':datalink,'selected_frames':selected}
    if transport == "local" and not missing:
        for p in input_paths:
            if not Path(p).exists():
                issues.append({"code": "missing_local_input", "severity": "error", "message": str(p)})
        for step in steps:
            if not (Path(repo) / step["argv"][1]).is_file():
                issues.append({"code": "missing_script", "severity": "error", "message": step["argv"][1]})
    value = {"schema": "pcm.execution-plan/v1", "created_at": utc_now(), "run_id": run_id,
             "kind": request["kind"], "version_name": version,
             "request_sha256": json_digest(request), "transport": transport, "ssh_alias": alias,
             "output_dir": output_dir, "effective_output_dir": effective,
             "input_paths": input_paths, "steps": steps, "missing_fields": missing,
             "issues": issues, "ready": not missing and not any(i["severity"] == "error" for i in issues),
             "source": request.get("source", {}), "dataset": dataset,
             "input_identity": identity,
             "scope": {"frame_fraction": fraction, "max_frames": evaluation.get("max_frames"),
                       "scene_names": evaluation.get("scene_names", [])}}
    value["plan_sha256"] = json_digest(value)
    if execution.get('ssh_options'):
        options = execution['ssh_options']
        validate_ssh_options(options)
        value['ssh_options'] = options
        value['scope']['expected_frames'] = evaluation.get('expected_frames')
        value['plan_sha256'] = json_digest({k:v for k,v in value.items() if k!='plan_sha256'})
    return value


def build_subnet_plan(request,run_id=None):
    """Frozen one-frame adapter for the original five v3.2.2 networks."""
    execution=request['execution'];model=request['model'];dataset=request['dataset'];scope=request['evaluation']
    run_id=run_id or 'pcm-v322-subnets-'+json_digest(request)[:16]
    if not re.fullmatch(r'(?:pcm-v322-subnets-|rm-)[A-Za-z0-9_-]{1,48}',run_id):raise ValueError('Invalid subnet run ID')
    if (request.get('version_name')!='v3.2.2' or dataset.get('class_name')!='SmartChassisDatasetV7'
            or type(scope.get('max_frames')) is not int or scope.get('max_frames')!=1
            or type(scope.get('expected_frames')) is not int or scope.get('expected_frames')!=1
            or scope.get('selection',{}).get('frames')!=[['2026-06-30_18-39-31_HLX34B167T1300224',18]]
            or scope.get('compute_gt_metrics') is not False):
        raise ValueError('This verified subnet adapter only accepts the frozen one-frame tensor smoke')
    alias=execution['ssh_alias']
    if not SAFE_ALIAS.fullmatch(alias):raise ValueError('Invalid SSH alias')
    identity={'files':{}};paths=[]
    def bind(path,sha):
        if not isinstance(path,str) or not PurePosixPath(path).is_absolute() or not re.fullmatch('[a-f0-9]{64}',str(sha)):
            raise ValueError('Subnet input requires an absolute remote path and SHA256')
        identity['files'][path]=sha;paths.append(path)
    bind(model['onnx'],model['onnx_sha256'])
    archives=model.get('subnet_archives')
    if not isinstance(archives,list) or len(archives)!=5 or {x.get('stage') for x in archives}!={0,1,2,3,4}:
        raise ValueError('All five subnet archives are required')
    for item in archives:bind(item['path'],item['sha256'])
    bind(dataset['data_link_file'],dataset['data_link_sha256'])
    for item in dataset['cached_inputs']:bind(str(PurePosixPath(dataset['input_dir'])/item['filename']),item['sha256'])
    bind(execution['driver_script'],execution['driver_sha256'])
    bind(execution['contract_script'],execution['contract_sha256'])
    bind(model['source_audit'],model['source_audit_sha256'])
    repo=execution['repo_dir'];output=str(PurePosixPath(execution['output_root'])/run_id)
    if not PurePosixPath(repo).is_absolute() or not PurePosixPath(output).is_absolute():raise ValueError('Absolute execution directories required')
    frozen=dict(request,run_id=run_id)
    value={'schema':'pcm.execution-plan/v1','created_at':utc_now(),'run_id':run_id,'kind':'compile_alignment',
        'version_name':'v3.2.2','request_sha256':json_digest(request),'transport':'ssh','ssh_alias':alias,
        'output_dir':output,'effective_output_dir':output,'input_paths':paths,
        'steps':[{'name':'compile_alignment','cwd':repo,'env':{},'argv':[execution['python'],execution['driver_script'],
                                                                    '--request-json',json.dumps(frozen,ensure_ascii=False)]}],
        'missing_fields':[],'issues':[],'ready':True,'source':request.get('source',{}),'dataset':dataset,
        'input_identity':identity,'scope':{'frame_fraction':1.0,'max_frames':1,'expected_frames':1,
            'selected_frames':scope['selection']['frames']}}
    value['plan_sha256']=json_digest(value)
    return value


def verify_plan(plan: dict) -> None:
    payload = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if plan.get("schema") != "pcm.execution-plan/v1" or json_digest(payload) != plan.get("plan_sha256"):
        raise ValueError("Plan digest does not match; regenerate the plan from its request")
    if not plan.get("ready"):
        raise ValueError("Plan has unresolved input/compatibility fields; see missing_fields and issues")


def validate_ssh_options(options):
    if not isinstance(options,dict) or set(options)-{'port','identity_file','user'}:
        raise ValueError('Only a configured SSH port and identity file are accepted')
    if 'port' in options and (type(options['port']) is not int or not 1<=options['port']<=65535):
        raise ValueError('Invalid SSH port')
    if 'user' in options and (not isinstance(options['user'],str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]{0,63}',options['user'])):
        raise ValueError('Invalid configured SSH user')
    if 'identity_file' in options:
        key=options['identity_file']
        if not isinstance(key,str) or not Path(key).is_absolute() or any(x in key for x in '\r\n\x00'):
            raise ValueError('SSH identity requires a configured absolute local path')


def ssh_command(alias: str, argv: list[str], cwd: str | None = None, env: dict | None = None, *, options=None) -> list[str]:
    if not SAFE_ALIAS.fullmatch(alias):
        raise ValueError("Invalid SSH alias")
    command = "exec " + ("env " + " ".join(shlex.quote(f"{k}={v}") for k, v in env.items()) + " " if env else "")
    command += shlex.join(argv)
    if cwd:
        command = "cd " + shlex.quote(cwd) + " && " + command
    # On Windows, prefer the native executable over .bat shims; argv must not
    # pass through cmd.exe quoting rules.
    executable = shutil.which("ssh.exe") if os.name == "nt" else shutil.which("ssh")
    if not executable:
        raise ValueError("An OpenSSH executable is required")
    extra=[]
    if options:
        validate_ssh_options(options)
        if options.get('port'): extra += ['-p',str(options['port'])]
        if options.get('user'): extra += ['-l', options['user']]
        if options.get('identity_file'): extra += ['-o','IdentitiesOnly=yes','-i',options['identity_file']]
    return [executable, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", *extra, alias, command]
