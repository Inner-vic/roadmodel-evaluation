"""Publish generated single-report XML once and retain unknown publication outcomes."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from .common import digest, load_json, save_json


def cli_command():
    cli = Path(shutil.which('lark-cli') or 'lark-cli')
    # Invoke npm's real Node entry point, avoiding Windows .cmd quoting wrappers.
    node = shutil.which('node.exe') or shutil.which('node')
    script = Path.home() / 'AppData/Roaming/npm/node_modules/@larksuite/cli/scripts/run.js'
    if node and script.is_file(): return [node, str(script)]
    return [str(cli)]


def portable_import(tree, source, root):
    project = Path(__file__).resolve().parents[1]
    for node in tree.iter():
        reference = node.get('path')
        if not reference or not reference.startswith('@./'): continue
        relative = reference[3:]
        candidates = [base / relative for base in (source.parent, project)]
        matches = [p.resolve() for p in candidates if p.is_file()]
        if not matches or len({digest(p) for p in matches}) != 1:
            raise ValueError('Report local resource missing or ambiguous')
        original = matches[0]
        filename = digest(original)[:16] + '-' + original.name
        dest = root / 'assets' / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, dest)
        node.set('path', '@./assets/' + filename)
    path = root / 'import.xml'
    path.write_text(''.join(ET.tostring(x,encoding='unicode') for x in tree),encoding='utf-8')
    return path


def publish_report(xml_file, output, *, invoke=None):
    source = Path(xml_file).resolve()
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt_file = root / 'publication.json'
    if receipt_file.exists():
        saved = load_json(receipt_file)
        if saved.get('source_sha256') != digest(source): raise ValueError('Publication source changed')
        if saved['state'] == 'SUCCEEDED': return saved
        raise ValueError('Publication outcome needs reconciliation; do not create another document')
    tree = ET.fromstring('<root>' + source.read_text(encoding='utf-8') + '</root>')
    titles = tree.findall('title')
    if len(titles) != 1: raise ValueError('Expected one report title')
    import_file = portable_import(tree,source,root)
    receipt = {'state': 'CREATING', 'source_sha256': digest(source), 'report_url': None}
    save_json(receipt_file, receipt)
    sequence = 0
    def call(args):
        nonlocal sequence
        if invoke: return invoke(args)
        sequence += 1
        proc = subprocess.run([*cli_command(), *args, '--as', 'user'], cwd=root,
                              capture_output=True, timeout=180, shell=False)
        (root / f'cli-{sequence}.stdout.json').write_bytes(proc.stdout)
        (root / f'cli-{sequence}.stderr.txt').write_bytes(proc.stderr)
        value = json.loads((proc.stdout or proc.stderr).decode('utf-8-sig'))
        if proc.returncode or not value.get('ok'): raise ValueError('Feishu report operation failed')
        return value['data']
    try:
        value = call(['docs', '+create', '--content', '@./' + import_file.name, '--doc-format', 'xml'])
        save_json(root / 'create.json', value)
        doc = value.get('document') or value
        token = doc.get('document_id') or doc.get('doc_id') or value.get('document_id') or value.get('doc_token')
        link = doc.get('url') or value.get('url') or value.get('document_url')
        if not token and link: token = link.rstrip('/').split('/')[-1]
        if not token: raise ValueError('Feishu document identity missing')
        receipt.update(document_id=token, report_url=link or 'https://li.feishu.cn/docx/' + token, state='VERIFYING')
        save_json(receipt_file, receipt, overwrite=True)
        if value.get('warnings'): raise ValueError('Feishu reported a partial import; inspect created document')
        after = call(['docs', '+fetch', '--doc', token, '--detail', 'full'])
        save_json(root / 'readback.json', after)
        actual = after['document']['content']
        original = ''.join(tree.itertext())
        # Compare substantive report headings/identity, not generated document block IDs.
        for node in [*titles, *tree.findall('h1')]:
            if ''.join(node.itertext()) not in ''.join(ET.fromstring('<root>'+actual+'</root>').itertext()):
                raise ValueError('Published report heading missing on readback')
        expected_tables = [[[''.join(cell.itertext()) for cell in row] for row in table.findall('.//tr')] for table in tree.iter('table')]
        actual_tree = ET.fromstring('<root>'+actual+'</root>')
        actual_tables = [[[''.join(cell.itertext()) for cell in row] for row in table.findall('.//tr')] for table in actual_tree.iter('table')]
        if expected_tables != actual_tables: raise ValueError('Published metric tables differ from generated report')
        for tag in ('source','img'):
            if len(list(actual_tree.iter(tag))) < len(list(tree.iter(tag))): raise ValueError('Published report resource missing')
        if not original.strip(): raise ValueError('Empty report')
        receipt['state'] = 'SUCCEEDED'
    except Exception:
        receipt['state'] = 'NEEDS_REVIEW'
        save_json(receipt_file, receipt, overwrite=True)
        raise
    save_json(receipt_file, receipt, overwrite=True)
    return receipt
