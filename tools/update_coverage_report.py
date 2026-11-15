"""Revision-bound updates of the same combined document; preserve old attachments."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import load_json, save_json, digest
from pcm_eval.publication import cli_command, portable_import

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/coverage-supplement-20261029'
DOC = 'DEMO00000000000000000000002'


def text(node): return ''.join(node.itertext())
def tables(root): return [[[text(c) for c in row] for row in t.findall('.//tr')] for t in root.iter('table')]


def main():
    pub = OUT / 'publication'
    pub.mkdir(exist_ok=False)
    sequence = 0
    journal = {'state': 'PREPARING', 'document_id': DOC, 'report_url': 'https://li.feishu.cn/docx/' + DOC}
    def checkpoint(state):
        journal['state'] = state
        save_json(pub / 'journal.json', journal, overwrite=True)
    def call(args):
        nonlocal sequence
        sequence += 1
        p = subprocess.run([*cli_command(), *args, '--as', 'user'], cwd=pub, capture_output=True, timeout=180)
        (pub / f'{sequence}.stdout.json').write_bytes(p.stdout)
        (pub / f'{sequence}.stderr.txt').write_bytes(p.stderr)
        d = json.loads((p.stdout or p.stderr).decode('utf-8-sig'))
        if p.returncode or not d.get('ok'): raise RuntimeError('Feishu operation failed; reconcile journal before any retry')
        data = d['data']
        if data.get('warnings') or data.get('result') in ('partial_success', 'failed'): raise RuntimeError('Partial write needs inspection')
        return data
    def read():
        d = call(['docs', '+fetch', '--doc', DOC, '--detail', 'full'])['document']
        return ET.fromstring('<root>' + d['content'] + '</root>'), d['revision_id']
    def bounds(root, title, level, stop):
        nodes = list(root)
        start = next(i for i, n in enumerate(nodes) if n.tag == level and text(n) == title)
        end = next((i for i in range(start + 1, len(nodes)) if stop(nodes[i])), len(nodes))
        return nodes[start:end]
    try:
        source = OUT / 'combined/report.xml'
        expected = ET.fromstring('<root>' + source.read_text(encoding='utf-8') + '</root>')
        before, revision = read()
        baseline = load_json(ROOT / 'evidence/reference-organized-20261029-02/publication/readback.json')['document']
        old = ET.fromstring('<root>' + baseline['content'] + '</root>')
        assert tables(before) == tables(old), 'Document tables changed since previous publication; preserve and review'
        assert len(tables(expected)) == len(tables(before)) == 5
        assert tables(expected)[0] == tables(before)[0], 'OCC must not change during vision/compile update'
        old_resources = {n.get('id') for n in before.iter('source') if n.get('id')}
        journal.update(start_revision=revision, original_source_blocks=sorted(old_resources))
        checkpoint('WRITING_VISION')
        prior = bounds(before, '视觉头', 'h2', lambda n: n.tag == 'h1')
        new = bounds(expected, '视觉头', 'h2', lambda n: n.tag == 'h1')
        fragment = pub / 'vision.xml'
        fragment.write_text(''.join(ET.tostring(n, encoding='unicode') for n in new), encoding='utf-8')
        changed = call(['docs', '+update', '--doc', DOC, '--command', 'block_replace', '--start-block-id', prior[0].get('id'),
                        '--end-block-id', prior[-1].get('id'), '--content', '@./vision.xml', '--doc-format', 'xml', '--revision-id', str(revision)])
        assert changed['result'] == 'success'
        current, revision = read()
        assert tables(current)[1:3] == tables(expected)[1:3]
        checkpoint('WRITING_COMPILE')
        stop = lambda n: n.tag == 'source' or any(x.tag == 'source' for x in n.iter())
        prior = bounds(current, '编译精度评测', 'h1', stop)
        new = bounds(expected, '编译精度评测', 'h1', stop)
        fragment = pub / 'compile.xml'
        fragment.write_text(''.join(ET.tostring(n, encoding='unicode') for n in new), encoding='utf-8')
        changed = call(['docs', '+update', '--doc', DOC, '--command', 'block_replace', '--start-block-id', prior[0].get('id'),
                        '--end-block-id', prior[-1].get('id'), '--content', '@./compile.xml', '--doc-format', 'xml', '--revision-id', str(revision)])
        assert changed['result'] == 'success'
        current, revision = read()
        assert tables(current) == tables(expected)
        paths = [OUT / p for p in ['vision-audit.json', 'compile-audit.json', 'mapping-audit.json', 'vision-selection.json', 'compile-selected.json',
            'vision-collected/execution_receipt.json', 'vision-collected/model-preflight.json', 'vision-collected/coverage_receipt.json',
            'vision-collected/normalized-result.json', 'compile-collected/cosine_summary.json', 'compile-collected/execution_receipt.json',
            'compile-collected/input_validation.json', 'compile-collected/output_integrity.json',
            'compile-collected/gt-adapted/metrics_board.json', 'compile-collected/gt-adapted/metrics_onnx.json',
            'compile-collected/gt-adapted/gt_receipt.json', 'compile-collected/gt-adapted/crosscheck.json', 'code/derivation.json']]
        paths.extend((OUT / 'vision-collected/output').glob('*/*.log'))
        bundle = OUT / 'verification-evidence.zip'
        manifest = [{'name': p.relative_to(OUT).as_posix(), 'sha256': digest(p), 'bytes': p.stat().st_size} for p in paths]
        with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED) as z:
            for p, entry in zip(paths, manifest): z.write(p, entry['name'])
            z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        appendix = ET.Element('root')
        note = ET.SubElement(appendix, 'p')
        note.text = '本次补覆盖更新的数据与核对材料如下；上一版附件保留供追溯。视觉4帧，编译3帧（复用1帧、新增2帧）。'
        ET.SubElement(appendix, 'source', path='@./data.json', name='当前补覆盖评测数据及来源.json')
        ET.SubElement(appendix, 'source', path='@./' + bundle.relative_to(ROOT).as_posix(), name='原始结果与逐项核对证据.zip')
        portable = portable_import(appendix, source, pub)
        checkpoint('APPENDING_EVIDENCE')
        changed = call(['docs', '+update', '--doc', DOC, '--command', 'append', '--content', '@./' + portable.name,
                        '--doc-format', 'xml', '--revision-id', str(revision)])
        assert changed['result'] == 'success'
        after, revision = read()
        assert tables(after) == tables(expected)
        assert old_resources.issubset({n.get('id') for n in after.iter('source')})
        assert len(list(after.iter('source'))) >= 3
        assert '本次GT未计算' not in text(after)
        journal.update(revision_id=revision, original_source_blocks_preserved=True, table_readback='all_five_exactly_match',
                       vision_subclass_cells_verified=56, compile_GT_cells_verified=26,
                       vision_frames=4, compile_frames=3, new_compile_inference_frames=2,
                       source_json_sha256=digest(OUT / 'combined/data.json'), evidence_zip_sha256=digest(bundle))
        checkpoint('VERIFIED')
        save_json(pub / 'verification.json', journal)
        print(json.dumps(journal, ensure_ascii=False, indent=2))
    except Exception:
        checkpoint('NEEDS_REVIEW')
        raise


if __name__ == '__main__':
    main()
