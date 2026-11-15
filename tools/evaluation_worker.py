"""Evaluate the two frozen jobs and queue publication for the requesting user."""
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcm_eval.common import json_digest, load_json, save_json
from pcm_eval.compile_preparation import prepare_compile
from pcm_eval.combined_delivery import write_combined
from pcm_eval.delivery_workflow import run_delivery, _write_atomic
from pcm_eval.evaluation import EvaluationRuntime
from pcm_eval.planning import build_plan


def evaluate(workflow, output, cfg, *, executor=None, reporter=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original = deepcopy(workflow)
    blocked = {}
    compiled = next(j for j in workflow['jobs'] if j['kind'] == 'compile_alignment')
    try:
        preparation = prepare_compile(compiled, output/'compile-preparation')
        if compiled.get('native_adapter') == 'compile_full_scope':
            from pcm_eval.compile_scope import prepare as prepare_scope
            preparation['scope_preparation'] = prepare_scope(compiled['plan'], output/'compile-scope', cfg['compile_profile'])
    except (ValueError, OSError) as error:
        preparation = {'state': 'BLOCKED', 'issues': [str(error)], 'bench_contacted': False, 'inference_frames': 0}
    if preparation['state'] != 'PREPARED':
        blocked['compile_alignment'] = list(dict.fromkeys(compiled['issues']+preparation.get('issues', [])))
    for job in workflow['jobs']:
        if job['kind'] == 'compile_alignment' and not cfg.get('bench_available', False):
            blocked.setdefault('compile_alignment', []).append('台架暂不可用，编译材料准备结果已保存')
        if job['kind'] == 'model_accuracy' and job['ready']:
            if job.get('native_adapter') == 'lpai_mounted_dataset':
                continue
            if job.get('environment') != 'lpai_devspace' or not workflow.get('dataset', {}).get('mount_receipt'):
                job['ready'] = False
                job['issues'].append('尚未核验本次输入对应的LPAI实际挂载')
            else:
                execution = job['request']['execution']
                space = cfg['model_space']
                execution['ssh_alias'] = space['host']
                execution['ssh_options'] = {k: space[k] for k in ('user', 'port', 'identity_file')}
                job['plan'] = build_plan(job['request'], job['plan']['run_id'])
    workflow.pop('workflow_sha256')
    workflow['workflow_sha256'] = json_digest(workflow)
    def execute(plan, target):
        if plan.get('schema') == 'roadmodel.lpai-native-plan/v1':
            from pcm_eval.lpai_model_transport import execute as execute_native
            return execute_native(plan, target, cfg['model_space'])
        if plan.get('schema') == 'roadmodel.compile-scope-plan/v1':
            if cfg.get('shared_hardware_queue', {}).get('enabled') is True:
                from pcm_eval.compile_hardware_queue import submit_and_wait
                return submit_and_wait(plan,target,cfg)
            from pcm_eval.compile_scope import execute as execute_scope
            return execute_scope(plan, target, cfg['compile_profile'], bench_available=cfg.get('bench_available') is True)
        from pcm_eval.runner import execute as execute_original
        return execute_original(plan, target)
    result = run_delivery(workflow, output/'effective', executor=executor or execute, reporter=reporter,
                          blocked_jobs=blocked)
    for chain in result['chains']:
        if chain['kind'] == 'compile_alignment':
            chain['preparation'] = preparation
            if (not cfg.get('bench_available', False) and preparation['state'] == 'PREPARED'
                    and chain['state'] not in ('SUCCEEDED', 'PARTIAL')):
                chain['state'] = 'WAITING_BENCH'
                chain['stage'] = '编译包与分网接口已核验' if preparation['state'] == 'PREPARED' else '编译材料准备待补齐'
    # The report is tied to the submitted workflow. Runtime gates do not change its identity.
    result['workflow_sha256'] = original['workflow_sha256']
    states = [c['state'] for c in result['chains']]
    if 'NEEDS_REVIEW' in states:
        result['state'] = 'NEEDS_REVIEW'
    elif 'WAITING_BENCH' in states:
        result['state'] = 'PARTIAL' if any(s in ('SUCCEEDED', 'PARTIAL') for s in states) else 'WAITING_BENCH'
    _write_atomic(output/'evaluation.json', result)
    return original, result


def main(run):
    run = Path(run).resolve()
    cfg = load_json(ROOT/'configs/evaluation.json')
    workflow = load_json(run/'workflow.json')
    if workflow['scope'] not in ('full', 'validation') or workflow['run_id'] != run.name:
        raise ValueError('只接受本工单冻结的完整评测计划')
    workflow, result = evaluate(workflow, run, cfg)
    report = write_combined(workflow, result, run/'combined/generation',
                            reference_file=cfg['reference_file'], template_file=ROOT/'configs/report_template.json')
    EvaluationRuntime(run.parents[1]).queue_publication(run.name, report, result)


if __name__ == '__main__':
    main(sys.argv[1])
