"""Run only the confirmed Demo's two frozen atomic jobs, then publish new reports."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcm_eval.common import json_digest, load_json
from pcm_eval.delivery_workflow import run_delivery, generate_report
from pcm_eval.planning import build_plan


def publish_demo_report(cfg, job, report, target):
    from pcm_eval.publication import publish_report
    def invoke(args):
        p = subprocess.run([cfg['lark_cli'], *args, '--as', 'bot'], cwd=target,
            env={**os.environ, 'LARKSUITE_CLI_CONFIG_DIR': cfg['lark_config'], 'LARKSUITE_CLI_NO_UPDATE_NOTIFIER': '1'},
            capture_output=True, timeout=180)
        response = json.loads(p.stdout or p.stderr)
        if p.returncode or response.get('ok') is not True:
            raise ValueError('报告发布待核验，不重复创建文档')
        return response['data']
    value = publish_report(report.get('report_xml') or report['xml'], target, invoke=invoke)
    if value['state'] != 'SUCCEEDED':
        raise ValueError('报告尚未通过飞书回读')
    return {**value, 'state': 'VERIFIED'}


def finish_combined(run, cfg, workflow, result, *, document_id=None):
    from pcm_eval.combined_delivery import write_combined
    from pcm_eval.combined_publication import publish_combined
    from pcm_eval.delivery_workflow import _write_atomic
    report = write_combined(workflow, result, run/'combined/generation',
                            reference_file=cfg['reference_file'], template_file=ROOT/'configs/report_template.json')
    target = run/'combined/publication'
    def invoke(args, *, cwd):
        p = subprocess.run([cfg['lark_cli'], *args, '--as', 'bot'], cwd=cwd,
            env={**os.environ, 'LARKSUITE_CLI_CONFIG_DIR': cfg['lark_config'], 'LARKSUITE_CLI_NO_UPDATE_NOTIFIER': '1'},
            capture_output=True, timeout=180)
        import uuid
        name = 'call-'+uuid.uuid4().hex[:12]
        target.mkdir(parents=True, exist_ok=True)
        (target/(name+'.stdout.json')).write_bytes(p.stdout)
        (target/(name+'.stderr.txt')).write_bytes(p.stderr)
        response = json.loads(p.stdout or p.stderr)
        if p.returncode or response.get('ok') is not True or response.get('data', {}).get('warnings'):
            raise ValueError('合并报告发布结果待核验，不新建另一份报告或重复评测')
        return response['data']
    try:
        publication = publish_combined(report, target, invoke=invoke, document_id=document_id)
        result['combined_report'] = {**report, **publication}
    except Exception as error:
        result.update(state='NEEDS_REVIEW', issues=['合并报告发布待核验：'+str(error)[:200]])
        _write_atomic(run/'progress.json', result)
        raise
    _write_atomic(run/'progress.json', result)
    return result


def main(run):
    run = Path(run).resolve()
    os.chdir('/mnt/data/bench-eval')
    cfg = load_json(ROOT/'configs/delivery_demo.json')
    workflow = load_json(run/'workflow.json')
    # Resource gates are installation configuration, never taken from a document.
    for job in workflow['jobs']:
        if job['kind'] == 'compile_alignment' and not cfg['bench_available']:
            job['ready'] = False
            job['issues'].append('等待台架可用；本Demo没有连接台架或派发编译推理。')
        if job['kind'] == 'model_accuracy' and job['ready']:
            request = job['request']
            request['execution']['ssh_alias'] = cfg['model_space']['host']
            request['execution']['ssh_options'] = {k: cfg['model_space'][k] for k in ('user', 'port', 'identity_file')}
            job['plan'] = build_plan(request, job['plan']['run_id'])
    workflow.pop('workflow_sha256')
    workflow['workflow_sha256'] = json_digest(workflow)
    # Keep the material-resolution snapshot separate from the effective resource gates.
    effective = run/'effective'
    effective.mkdir(exist_ok=True)
    def reporter(job, native, target):
        report = generate_report(job, native, target)
        if job['kind'] == 'model_accuracy':
            from pcm_eval.strict_model_template import model_tables, render
            data = load_json(report['report_json'])
            reference = load_json(cfg['reference_file'])['data']['document']['content']
            tables = model_tables(data, reference, load_json(ROOT/'configs/report_template.json')['model_accuracy'])
            body = render(reference, tables, model_version=job['version'], vision_samples=0)
            Path(report['report_xml']).write_text('<title>PCM 单次报告 - 模型评测 '+job['version']+'</title>'+body, encoding='utf-8')
        return report
    result = run_delivery(workflow, effective, reporter=reporter)
    finish_combined(run, cfg, workflow, result)


if __name__ == '__main__':
    main(sys.argv[1])
