"""Result contract checks shared by intake, delivery and report generation."""
from __future__ import annotations

import math
from pathlib import Path

from .common import digest, load_json

SCHEMA = Path(__file__).resolve().parents[1] / 'schemas/single_eval.schema.json'


def validate_result(result: dict, *, verify_artifacts: bool = True) -> dict:
    spec = load_json(SCHEMA)
    issues = []
    def issue(path, message):
        issues.append({'path': path, 'message': message})
    if not isinstance(result, dict):
        return {'valid': False, 'issues': [{'path': '', 'message': 'Result must be an object'}]}
    for key in spec['required']:
        if key not in result:
            issue(key, 'Required field is absent')
    if result.get('schema') != 'pcm.single-eval/v1':
        issue('schema', 'Unsupported result schema')
    if result.get('evaluation_type') not in ('model_accuracy', 'compile_alignment'):
        issue('evaluation_type', 'Unknown evaluation type')
    for key in ('execution', 'collection', 'coverage', 'provenance'):
        if not isinstance(result.get(key, {}), dict):
            issue(key, 'Expected an object')
    execution = result.get('execution') if isinstance(result.get('execution'), dict) else {}
    collection = result.get('collection') if isinstance(result.get('collection'), dict) else {}
    coverage = result.get('coverage') if isinstance(result.get('coverage'), dict) else {}
    if not isinstance(execution.get('state'), str):
        issue('execution/state', 'Execution state is required')
    if collection.get('status') not in ('complete', 'partial', 'empty'):
        issue('collection/status', 'Unknown collection status')
    if not isinstance(collection.get('issues'), list):
        issue('collection/issues', 'Expected issues array')
    expected = coverage.get('expected_frames')
    if expected is not None and (type(expected) is not int or expected < 1):
        issue('coverage/expected_frames', 'Expected a positive integer or null')
    observed = coverage.get('observed', {})
    if not isinstance(observed, dict):
        issue('coverage/observed', 'Expected an object')
    else:
        for key, value in observed.items():
            if type(value) is not int or value < 0:
                issue('coverage/observed/' + key, 'Expected a nonnegative integer')
            elif expected and key in ('model_samples', 'tensor_comparison', 'board_metrics', 'onnx_metrics') and value != expected:
                issue('coverage/observed/' + key, 'Actual coverage differs from requested scope')
    units = spec['properties']['metrics']['items']['properties']['unit']['enum']
    metrics = result.get('metrics', [])
    if not isinstance(metrics, list):
        issue('metrics', 'Expected an array')
        metrics = []
    identities = set()
    for i, metric in enumerate(metrics):
        path = f'metrics/{i}'
        if not isinstance(metric, dict):
            issue(path, 'Expected an object')
            continue
        for field in ('name', 'value', 'unit', 'dimensions', 'source'):
            if field not in metric:
                issue(path + '/' + field, 'Required field is absent')
        value = metric.get('value')
        if type(value) not in (int, float) or not math.isfinite(value):
            issue(path + '/value', 'Expected a finite numeric value; unavailable metrics are not zero-filled')
        if metric.get('unit') not in units:
            issue(path + '/unit', 'Unsupported metric unit')
        dimensions = metric.get('dimensions')
        if not isinstance(dimensions, dict):
            issue(path + '/dimensions', 'Expected an object')
        else:
            import json
            identity = (metric.get('name'), json.dumps(dimensions, sort_keys=True))
            if identity in identities:
                issue(path, 'Duplicate metric and dimensions')
            identities.add(identity)
        source = metric.get('source')
        if not isinstance(source, dict) or not all(isinstance(source.get(k), str) for k in ('path', 'locator')):
            issue(path + '/source', 'Source path and locator are required')
    artifacts = result.get('artifacts', [])
    if not isinstance(artifacts, list):
        issue('artifacts', 'Expected an array')
        artifacts = []
    import re
    seen = set()
    for i, item in enumerate(artifacts):
        path = f'artifacts/{i}'
        if not isinstance(item, dict) or not isinstance(item.get('path'), str):
            issue(path, 'Source file path is required')
            continue
        if item['path'] in seen:
            issue(path, 'Duplicate artifact path')
        seen.add(item['path'])
        if not isinstance(item.get('role'), str) or not item['role']:
            issue(path + '/role', 'Artifact role is required')
        if type(item.get('size_bytes')) is not int or item['size_bytes'] < 0:
            issue(path + '/size_bytes', 'Expected a nonnegative integer')
        if not re.fullmatch('[a-f0-9]{64}', str(item.get('sha256', ''))):
            issue(path + '/sha256', 'SHA256 is required')
        if verify_artifacts:
            source = Path(item['path'])
            if not source.is_file():
                issue(path, 'Referenced artifact is missing')
            elif source.stat().st_size != item.get('size_bytes') or digest(source) != item.get('sha256'):
                issue(path, 'Referenced artifact changed')
    if result.get('quality_assessment') != 'not_assessed':
        issue('quality_assessment', 'This contract does not make a release decision')
    return {'valid': not issues, 'validation': 'roadmodel_result_contract', 'issues': issues,
            'metric_count': len(metrics), 'artifact_count': len(artifacts)}


def require_result(result: dict, *, verify_artifacts: bool = True) -> dict:
    check = validate_result(result, verify_artifacts=verify_artifacts)
    if not check['valid']:
        raise ValueError('Result contract failed: ' + '; '.join(x['path'] + ': ' + x['message'] for x in check['issues'][:8]))
    return check
