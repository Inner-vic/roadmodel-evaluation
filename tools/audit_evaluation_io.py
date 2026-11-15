"""Offline audit of the existing, actually exercised evaluation boundaries."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/io-contract-audit-20261029'
SOURCES = {
    'model_occ': 'evidence/model-evaluation/native-v3.2.2-smoke-20261027/normalized_result.json',
    'model_vision': 'evidence/model-evaluation/remaining-heads-20261027/collected/vision-01/normalized_result-reviewed.json',
    'model_trajectory': 'evidence/model-evaluation/remaining-heads-20261027/collected/trajectory-01-recovered/normalized_result-reviewed.json',
    'compile_with_gt': 'evidence/actual-runs/pcm-v7-multiscene-001/gt-adapted-20261027/result.json',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['before', 'current'], default='current')
    args = parser.parse_args()
    schema_path = ROOT / 'schemas/single_eval.schema.json'
    schema = json.loads(schema_path.read_text(encoding='utf-8'))
    try:
        from jsonschema import Draft202012Validator
        validator = Draft202012Validator(schema)
    except ImportError:
        validator = None
    rows = []
    for name, relative in SOURCES.items():
        path = ROOT / relative
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        errors = []
        if validator:
            for error in validator.iter_errors(data):
                errors.append({'path': '/'.join(map(str, error.absolute_path)), 'message': error.message})
        else:
            # Explicitly report reduced validation instead of claiming JSON Schema compliance.
            errors.extend({'path': key, 'message': 'Required field is absent'}
                          for key in schema['required'] if key not in data)
            units = schema['properties']['metrics']['items']['properties']['unit']['enum']
            errors.extend({'path': f'metrics/{i}/unit', 'message': 'Unit absent from declared schema: ' + str(m.get('unit'))}
                          for i, m in enumerate(data['metrics']) if m.get('unit') not in units)
        missing = []
        mismatches = []
        oversized = []
        for artifact in data['artifacts']:
            source = Path(artifact['path'])
            if not source.is_file():
                missing.append(str(source))
            elif source.stat().st_size > 64 * 1024 * 1024:
                oversized.append(str(source))
            elif source.stat().st_size != artifact['size_bytes'] or sha(source) != artifact['sha256']:
                mismatches.append(str(source))
        rows.append({'case': name, 'source': relative, 'source_sha256': sha(path),
                     'evaluation_type': data['evaluation_type'], 'version_name': data['version_name'],
                     'execution': data['execution'], 'collection_status': data['collection']['status'],
                     'coverage': data['coverage'], 'metrics': len(data['metrics']),
                     'units': sorted({m['unit'] for m in data['metrics']}),
                     'declared_artifacts': len(data['artifacts']), 'schema_errors': errors,
                     'missing_artifacts': missing, 'mismatched_artifacts': mismatches,
                     'large_artifacts_not_rehashed': oversized})
    value = {'schema': 'roadmodel.io-contract-audit/v1', 'mode': 'existing_local_evidence_only',
             'new_inference_frames': 0, 'network_requests': 0,
             'validation': 'jsonschema.draft202012' if validator else 'required_fields_and_units_only',
             'schema_sha256': sha(schema_path), 'cases': rows}
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / ('audit' + ('-before' if args.phase == 'before' else '') + ('' if validator else '-reduced') + '.json')
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'validation': value['validation'],
                      'cases': [{k: row[k] for k in ['case', 'metrics', 'declared_artifacts', 'schema_errors',
                                 'missing_artifacts', 'mismatched_artifacts']} for row in rows]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
