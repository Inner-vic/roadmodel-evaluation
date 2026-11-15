"""Existing results only: five original PCM tables in a combined presentation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcm_eval.common import load_json, save_json
from pcm_eval.model_report import build_model_report
from pcm_eval.reference_report import write_reference_report
from pcm_eval.validation import require_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--draft', required=True)
    args = parser.parse_args()
    base = Path(args.output)
    if base.exists():
        raise ValueError('Choose a new output directory')
    occ = ROOT / 'evidence/feishu-entry-20261029/rm-small-20261029-01/model_accuracy/reviewed-report-02/generated/report.json'
    previous = load_json(occ)
    heads = [load_json(ROOT / ('evidence/model-evaluation/remaining-heads-20261027/collected/' + p)) for p in
             ['vision-01/normalized_result-reviewed.json', 'trajectory-01-recovered/normalized_result-reviewed.json']]
    for result in [previous['source_result'], *heads]:
        require_result(result)
    model = build_model_report(previous['source_result'], binding=previous['binding'],
                               gt_preparation=previous['gt_preparation'], head_results=heads)
    save_json(base / 'bound-model.json', model)
    data = write_reference_report(base / 'combined', model_report=base / 'bound-model.json',
                compile_report=ROOT / 'evidence/compile-subnet-review-20261029/pcm-v322-subnets-one-02/reviewed-delivery/report/report.json',
                reference_file=ROOT / 'evidence/source-docs/release-template-review-20261029.json')
    # Authoring draft refers to the attachment relative to the project CLI CWD.
    xml = (base / 'combined/report.xml').read_text(encoding='utf-8')
    reference = (base / 'combined/data.json').resolve().relative_to(ROOT).as_posix()
    xml = xml.replace('@./data.json', '@./' + reference)
    Path(args.draft).write_text(xml, encoding='utf-8')
    print(json.dumps({'state': 'organized', 'tables': len(data['tables']), 'new_inference_frames': 0,
                      'scopes': data['sample_scopes'], 'occ_gt_support': data['occ_gt_support']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
