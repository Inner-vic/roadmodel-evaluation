"""Create one document per task; update that document when either chain changes."""
from pathlib import Path
import xml.etree.ElementTree as ET

from .common import digest, load_json, save_json
from .publication import publish_report
from .feishu_update import table_values


def publish_combined(report, output, *, invoke, document_id=None):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    source = Path(report['report_xml'])
    data = load_json(report['report_json'])
    if data.get('schema') != 'roadmodel.combined-delivery/v1':
        raise ValueError('需要本次双链路合并报告')
    binding = root/'document.json'
    identity = {'task_sha256': data['task_sha256']}
    provided = invoke
    invoke = lambda args: provided(args, cwd=root)
    if binding.exists():
        bound = load_json(binding)
        if bound['task_sha256'] != identity['task_sha256']:
            raise ValueError('同一报告链接不能绑定不同任务')
        if document_id and document_id != bound['document_id']:
            raise ValueError('不能更换本次报告文档')
        document_id = bound['document_id']
    elif document_id:
        save_json(binding, {**identity, 'document_id': document_id})
    version = root/'versions'/report['source_sha256']
    proof = version/'verification.json'
    if proof.exists():
        value = load_json(proof)
        if value.get('task_sha256', identity['task_sha256']) != identity['task_sha256']:
            raise ValueError('已有发布收据的任务身份不一致')
        if value.get('workflow_sha256', data['workflow_sha256']) != data['workflow_sha256']:
            raise ValueError('已有发布收据的工作流身份不一致')
        return {**value, **identity, 'workflow_sha256': data['workflow_sha256']}
    version.mkdir(parents=True, exist_ok=True)
    if not document_id:
        # A previous create must be reconciled before a changed report can create anything.
        for prior in (root/'versions').glob('*/created/publication.json'):
            value = load_json(prior)
            if value['state'] != 'SUCCEEDED':
                raise ValueError('原合并报告创建结果待核验，不重复创建文档')
            document_id = value['document_id']
        if not document_id:
            created = publish_report(source, version/'created', invoke=lambda args: provided(args, cwd=version/'created'))
            document_id = created['document_id']
            save_json(binding, {**identity, 'document_id': document_id})
            result = {**created, **identity, 'state': 'VERIFIED', 'table_count': 5,
                      'all_metric_cells_match': True, 'all_evaluations_complete': data['all_evaluations_complete'],
                      'workflow_sha256': data['workflow_sha256']}
            save_json(proof, result)
            return result
        save_json(binding, {**identity, 'document_id': document_id})
    journal = version/'journal.json'
    if journal.exists():
        raise ValueError('本次报告更新结果待核验，不重复写入或新建文档')
    expected = ET.fromstring('<root>'+source.read_text(encoding='utf-8')+'</root>')
    expected_tables = table_values(expected)
    if len(expected_tables) != 5:
        raise ValueError('合并报告必须包含五张原模板指标表')
    before = invoke(['docs', '+fetch', '--doc', document_id, '--detail', 'full'])['document']
    save_json(version/'before.json', before)
    tree = ET.fromstring('<root>'+before['content']+'</root>')
    nodes = [n for n in tree if n.tag != 'title']
    tail = next((i for i, n in enumerate(nodes) if any(c.tag in ('source', 'img', 'whiteboard', 'sheet', 'bitable') for c in n.iter())), len(nodes))
    body = nodes[:tail]
    if not body or any(n.tag == 'table' for n in nodes[tail:]):
        raise ValueError('已有报告正文与资源交错，需核验后再更新')
    keep = {n.get('token') or n.get('id') for n in tree.iter('source')}
    contents = ''.join(ET.tostring(n, encoding='unicode') for n in expected if n.tag != 'title')
    path = version/'body.xml'
    path.write_text(contents, encoding='utf-8')
    save_json(journal, {'state': 'WRITING', **identity, 'document_id': document_id,
                        'source_sha256': digest(source), 'start_revision': before['revision_id']})
    # invoke resolves local @paths relative to its configured CWD.
    invoke(['docs', '+update', '--doc', document_id, '--command', 'block_replace',
            '--start-block-id', body[0].get('id'), '--end-block-id', body[-1].get('id'),
            '--content', '@./'+path.relative_to(root).as_posix(), '--doc-format', 'xml',
            '--revision-id', str(before['revision_id'])])
    title = ''.join(expected.find('title').itertext())
    previous_title = tree.find('title')
    if previous_title is None or ''.join(previous_title.itertext()) != title:
        invoke(['drive', '+update-title', '--url', 'https://li.feishu.cn/docx/'+document_id, '--title', title])
    after = invoke(['docs', '+fetch', '--doc', document_id, '--detail', 'full'])['document']
    save_json(version/'after.json', after)
    actual = ET.fromstring('<root>'+after['content']+'</root>')
    if table_values(actual) != expected_tables:
        raise ValueError('合并报告指标表回读不一致')
    if not keep.issubset({n.get('token') or n.get('id') for n in actual.iter('source')}):
        raise ValueError('已有附件未保留')
    text = ''.join(actual.itertext())
    for node in list(expected):
        value = ''.join(node.itertext())
        if (node.tag in ('h1', 'title') or ('尚未完成' in value and node.tag == 'p')) and value not in text:
            raise ValueError('合并报告标题或待完成状态回读缺失')
    result = {'state': 'VERIFIED', 'document_id': document_id,
              'report_url': 'https://li.feishu.cn/docx/'+document_id, 'revision_id': after['revision_id'],
              'table_count': 5, 'all_metric_cells_match': True,
              'all_evaluations_complete': data['all_evaluations_complete'],
              **identity, 'workflow_sha256': data['workflow_sha256'], 'new_inference_frames': 0}
    save_json(proof, result)
    return result
