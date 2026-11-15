"""Exercise existing report generators using retained results, without inference."""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcm_eval.compile_report import write_compile_report
from pcm_eval.model_report import write_model_report
from pcm_eval.common import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-existing', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'evidence/io-contract-audit-20261029/report-replay'
    if args.verify_existing:
        receipt = out / 'replay-receipt.json'
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        model = saved['model']
        compile_report = saved['compile']
        for item in saved['top_level_inputs']:
            assert digest(ROOT / item['path']) == item['sha256']
        for path in [model['report_xml'], compile_report['xml']]:
            ET.fromstring('<root>' + Path(path).read_text(encoding='utf-8') + '</root>')
        for path in [model['evidence_bundle'], compile_report['details']]:
            with zipfile.ZipFile(path) as archive:
                assert archive.testzip() is None
        model_data = json.loads(Path(model['report_json']).read_text(encoding='utf-8'))
        compile_data = json.loads(Path(compile_report['json']).read_text(encoding='utf-8'))
        assert len(model_data['all_metrics']) == 426
        assert compile_data['coverage']['actual_frames'] == 20
        value = {'state': 'passed', 'xml_documents_parsed': 2, 'zip_integrity_checked': 2,
                 'input_json_hashes_checked': len(saved['top_level_inputs']),
                 'new_inference_frames': 0, 'network_requests': 0}
        (out / 'replay-verification.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(value))
        return
    if out.exists():
        raise ValueError('Replay output already exists; preserve previous evidence')
    occ = ROOT / 'evidence/model-evaluation/native-v3.2.2-smoke-20261027/normalized_result.json'
    binding = ROOT / 'evidence/model-evaluation/model-reference-20261027/model-input-binding.json'
    gt = ROOT / 'evidence/model-evaluation/model-reference-20261027/native-gt-preparation.json'
    heads = [ROOT / 'evidence/model-evaluation/remaining-heads-20261027/collected' / name / 'normalized_result-reviewed.json'
             for name in ['vision-01', 'trajectory-01-recovered']]
    compile_result = ROOT / 'evidence/actual-runs/pcm-v7-multiscene-001/gt-adapted-20261027/result.json'
    model = write_model_report(out / 'model', result_file=occ, binding_file=binding,
                               gt_preparation_file=gt, head_result_files=heads)
    compile_report = write_compile_report(out / 'compile/report.xml', result_file=compile_result)
    model_data = json.loads(Path(model['report_json']).read_text(encoding='utf-8'))
    compile_data = json.loads(Path(compile_report['json']).read_text(encoding='utf-8'))
    assert len(model_data['all_metrics']) == 426
    assert compile_data['coverage']['actual_frames'] == 20
    assert model['new_inference_frames'] == compile_report['new_inference_frames'] == 0
    for path in [model['report_xml'], compile_report['xml']]:
        ET.fromstring('<root>' + Path(path).read_text(encoding='utf-8') + '</root>')
    for path in [model['evidence_bundle'], compile_report['details']]:
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
    value = {'schema': 'roadmodel.io-report-replay/v1', 'state': 'passed',
             'new_inference_frames': 0, 'network_requests': 0,
             'layout': 'existing generators; proposed business layout not yet implemented',
             'model': model, 'compile': compile_report,
             'top_level_inputs': [{'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}
                                  for path in [occ, binding, gt, *heads, compile_result]],
             'dependency_note': 'Referenced result artifacts and compile sidecars were also read; the listed JSON files are not standalone portable bundles.'}
    receipt = out / 'replay-receipt.json'
    receipt.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
