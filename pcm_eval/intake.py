"""Resolve explicit submissions into independently runnable/replayable jobs."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .common import digest, json_digest, load_json, save_json, utc_now
from .planning import KINDS, build_plan
from .validation import validate_result

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = ROOT / 'configs/evaluation_profiles.json'


def resolve_submission(submission, *, profile_file=None, base_dir=None):
    base = Path(base_dir or ROOT).resolve()
    registry = load_json(profile_file or DEFAULT_PROFILES)
    if not isinstance(submission,dict) or submission.get('schema') != 'roadmodel.submission/v1':
        raise ValueError('Use roadmodel.submission/v1')
    action = submission.get('action', 'report_existing')
    if action not in ('report_existing', 'prepare_execution'):
        raise ValueError('Action must be report_existing or prepare_execution; execution is separate')
    jobs = submission.get('jobs')
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('Provide at least one independent evaluation job')
    resolved, all_issues, ids = [], [], set()
    def local(value):
        p = Path(value)
        return p.resolve() if p.is_absolute() else (base / p).resolve()
    for i, job in enumerate(jobs):
        if not isinstance(job,dict):raise ValueError('Each evaluation job must be an object')
        unknown=set(job)-{'id','profile','kind','model_ref','dataset_ref','report','request_file','scope','run_id','layout'}
        if unknown:raise ValueError('Unknown job fields: '+', '.join(sorted(unknown)))
        issues = []
        name = job.get('id', f'job-{i + 1}')
        import re
        if not isinstance(name, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name) or name in ids:
            raise ValueError('Job IDs must be unique safe names')
        ids.add(name)
        profile_name = job.get('profile')
        if not profile_name and job.get('model_ref'):
            candidates = [(key, value) for key, value in registry.get('profiles', {}).items()
                          if value.get('model_ref') == job['model_ref']
                          and (not job.get('kind') or value.get('kind') == job['kind'])]
            if len(candidates) == 1:
                profile_name = candidates[0][0]
            elif len(candidates) > 1:
                issues.append({'code': 'ambiguous_model_binding', 'field': 'model_ref', 'message': 'Several verified bindings match; specify one profile'})
        profile = registry.get('profiles', {}).get(profile_name)
        if job.get('profile') and not profile:
            issues.append({'code': 'unknown_profile', 'field': 'profile', 'message': 'Unknown profile; no model version will be guessed'})
        profile = deepcopy(profile or {})
        kind = job.get('kind', profile.get('kind'))
        if kind not in KINDS:
            issues.append({'code': 'missing_kind', 'field': 'kind', 'message': 'Choose model_accuracy or compile_alignment'})
        if profile.get('kind') and kind != profile['kind']:
            issues.append({'code': 'kind_mismatch', 'field': 'kind', 'message': 'Profile evaluation type differs'})
        for field in ('model_ref', 'dataset_ref'):
            if job.get(field) and job[field] != profile.get(field):
                issues.append({'code': 'unbound_reference', 'field': field,
                               'message': 'Reference has no verified binding in this profile; provide an explicit request/report manifest'})
        item = {'id': name, 'kind': kind, 'profile': profile_name, 'action': action, 'issues': issues}
        if action == 'report_existing':
            if job.get('scope'):
                issues.append({'code':'existing_scope_frozen','field':'scope','message':'Report replay preserves the measured scope; use prepare_execution to select new samples'})
            config = deepcopy(job.get('report') or profile.get('report') or {})
            result_files = [config.get('result'), *config.get('head_results', [])]
            if not result_files[0]:
                issues.append({'code': 'missing_result', 'field': 'report.result', 'message': 'One existing matching result is required'})
            checks = []
            inputs = {}
            for field in ('result', 'binding_file', 'gt_preparation_file', 'request_file', 'source_dir', 'supplemental_dir', 'trajectory_figure_file'):
                if config.get(field):
                    path = local(config[field])
                    inputs[field] = str(path)
                    if not path.exists():
                        issues.append({'code': 'missing_input', 'field': field, 'message': str(path)})
            inputs['head_results'] = [str(local(p)) for p in config.get('head_results', [])]
            for source in result_files:
                if not source or not local(source).is_file():
                    continue
                result = load_json(local(source))
                check = validate_result(result)
                checks.append({'file': str(local(source)), 'sha256': digest(local(source)), **check})
                if result.get('evaluation_type') != kind:
                    issues.append({'code': 'wrong_result_type', 'field': 'result', 'message': str(source)})
                if not check['valid']:
                    issues.append({'code': 'invalid_result', 'field': 'result', 'message': check['issues']})
                if result.get('execution', {}).get('state') != 'succeeded' or result.get('collection', {}).get('status') != 'complete':
                    issues.append({'code': 'incomplete_result', 'field': 'result', 'message': 'This workflow publishes only successfully collected results'})
            for source in inputs['head_results']:
                if not Path(source).is_file():
                    issues.append({'code': 'missing_head_result', 'field': 'head_results', 'message': source})
            item.update(inputs=inputs, checks=checks, layout=job.get('layout', 'business'))
            if item['layout'] not in ('business', 'diagnostic'):
                issues.append({'code': 'unknown_layout', 'field': 'layout', 'message': 'Choose business or diagnostic'})
        else:
            item['execution_heads']=profile.get('execution_heads',[])
            request_file = job.get('request_file', profile.get('request_file'))
            if request_file and local(request_file).is_file():
                request = load_json(local(request_file))
                if job.get('scope'):
                    if not isinstance(job['scope'],dict) or set(job['scope'])-{'scene_names','max_frames','expected_frames','frame_fraction','selection','temporal'}:
                        raise ValueError('Scope may select samples, not override model/input verification settings')
                    request.setdefault('evaluation', {}).update(job['scope'])
                evaluation=request.get('evaluation',{})
                if evaluation.get('expected_frames') and evaluation.get('max_frames') and evaluation['expected_frames']!=evaluation['max_frames']:
                    issues.append({'code':'scope_count_mismatch','field':'scope','message':'expected_frames and max_frames disagree'})
                if kind != request.get('kind'):
                    issues.append({'code': 'request_kind_mismatch', 'field': 'request_file', 'message': 'Request type differs'})
                plan = build_plan(request, job.get('run_id'))
                item.update(request=request, plan=plan)
                issues.extend({'code': 'missing_field', 'field': p, 'message': 'Resolved execution input is required'} for p in plan['missing_fields'])
                issues.extend(plan['issues'])
            else:
                issues.append({'code': 'missing_execution_binding', 'field': 'request_file',
                               'message': 'Supply a fully bound execution request; retained model launch scripts are not silently rerun'})
        item['ready'] = not issues
        resolved.append(item)
        all_issues.extend(dict(issue, job_id=name) for issue in issues)
    return {'schema': 'roadmodel.resolved-submission/v1', 'created_at': utc_now(),
            'submission_sha256': json_digest(submission), 'profile_file_sha256': digest(Path(profile_file or DEFAULT_PROFILES)),
            'ready': not all_issues, 'action': action, 'jobs': resolved, 'issues': all_issues,
            'execution_started': False, 'publication_started': False}


def write_resolution(submission_file, output, *, profile_file=None):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Resolution output already exists; preserve prior plans')
    value = resolve_submission(load_json(submission_file), profile_file=profile_file)
    plans=output.parent/(output.stem+'.jobs')
    for job in value['jobs']:
        if job['ready'] and job.get('plan'):
            plan_file=plans/(job['id']+'.plan.json')
            save_json(plan_file,job['plan'])
            job['plan_file']=str(plan_file)
    save_json(output, value)
    return value
