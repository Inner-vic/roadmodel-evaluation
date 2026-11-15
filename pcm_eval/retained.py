"""Report-only reuse of an administrator-verified, exact model/dataset binding.

The user cannot submit a result path or a report URL. Completed inference is never
dispatched by this adapter; original run identity and coverage remain unchanged.
"""
from pathlib import Path
from .common import load_json, digest, save_json
from .validation import require_result


def checked_file(root, record):
    root = Path(root).resolve()
    path = (root / record['path']).resolve()
    path.relative_to(root)
    if not path.is_file() or digest(path) != record['sha256']:
        raise ValueError('Registered retained artifact is missing or changed')
    return path


def verify(root, record, kind, version, dataset_id):
    receipt_path = checked_file(root, record)
    receipt = load_json(receipt_path)
    if (receipt.get('schema') != 'roadmodel.retained-binding/v1' or receipt.get('state') != 'VERIFIED'
            or receipt.get('kind') != kind or receipt.get('version') != version
            or receipt.get('dataset_id') != dataset_id or receipt.get('new_inference_frames') != 0):
        raise ValueError('Retained result identity is inconsistent with submitted materials')
    report_path = checked_file(root, receipt['atomic_report'])
    report = load_json(report_path)
    result = report['source_result'] if kind == 'model_accuracy' else report['source_normalized_result']
    require_result(result, verify_artifacts=False)
    if result['execution']['state'] != 'succeeded' or result['collection']['status'] != 'complete':
        raise ValueError('Only complete, successful retained evaluations can be reused')
    report_version = report.get('version_name') if kind == 'model_accuracy' else report['identity']['model_version']
    if report_version != version:
        raise ValueError('Retained report version mismatch')
    for item in receipt['evidence']:
        checked_file(root, item)
    return receipt_path, receipt, report_path, report


def execute(job, target):
    root = Path(job['material_root'])
    receipt_path, receipt, source, report = verify(root, job['retained_result'], job['kind'], job['version'], job['dataset_id'])
    target.mkdir(parents=True, exist_ok=False)
    save_json(target / 'report.json', report)
    if job['kind'] == 'model_accuracy':
        from .model_report import render_model_xml
        xml = render_model_xml(report, json_attachment='report.json')
    else:
        from .compile_report import render_sections
        from .business_layout import compile_business_sections
        sections = compile_business_sections(report, render_sections(report))
        xml = '<title>编译精度单次评测报告 '+job['version']+'</title>'+''.join(s['xml'] for s in sections)
    (target / 'report.xml').write_text(xml, encoding='utf-8')
    output = {'state': 'SUCCEEDED', 'mode': 'reuse_verified_results', 'new_inference_frames': 0,
              'source_binding_sha256': digest(receipt_path), 'source_report_sha256': digest(source),
              'atomic_report': str(target / 'report.json'), 'source_run_id': receipt['source_run_id'],
              'scope': receipt['scope'], 'issues': ['复用已验证的小场景结果，新增推理0帧。']}
    save_json(target / 'reuse-receipt.json', output)
    return output


def compose(run, details, *, root, delivery_record):
    """Regenerate the five existing tables and return their verified Feishu document."""
    from .reference_report import organize, render
    root = Path(root)
    delivery = load_json(checked_file(root, delivery_record))
    original = load_json(checked_file(root, delivery['combined_data']))
    reference = load_json(checked_file(root, delivery['reference']))
    publication = load_json(checked_file(root, delivery['publication']))
    if publication.get('state') != 'VERIFIED' or publication.get('table_readback') != 'all_five_exactly_match':
        raise ValueError('Original publication has no verified table readback')
    if publication['source_json_sha256'] != delivery['combined_data']['sha256']:
        raise ValueError('Publication was generated from different source data')
    model, compiled = [load_json(details[k]['atomic_report']) for k in ('model_accuracy', 'compile_alignment')]
    data = organize(model, compiled, reference)
    if data['tables'] != original['tables'] or data['sample_scopes'] != original['sample_scopes']:
        raise ValueError('Regenerated metrics or scope differ from published report')
    save_json(run / 'combined/data.json', data)
    (run / 'combined/report.xml').write_text(render(data, 'data.json'), encoding='utf-8')
    value = {'state':'SUCCEEDED', 'mode':'reuse_verified_publication', 'new_inference_frames':0,
             'report_url':publication['report_url'], 'document_id':publication['document_id'],
             'sample_scopes':data['sample_scopes'], 'five_tables_match':True,
             'original_publication_revision':publication['revision_id']}
    save_json(run / 'combined/delivery.json', value)
    return value
