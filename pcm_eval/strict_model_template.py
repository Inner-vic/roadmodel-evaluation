"""Project the supplied PCM template to one model run; add no prose."""
from copy import deepcopy
import re
import xml.etree.ElementTree as ET

from .business_layout import table
from .reference_report import display


def model_tables(model, reference_xml, spec):
    """Map one model result to the original three tables, without a compile input."""
    from .reference_report import cell, metric
    root=ET.fromstring('<root>'+reference_xml+'</root>')
    originals=list(root.iter('table'))[:3]
    labels=[[[''.join(c.itertext()) for c in row] for row in t.iter('tr')] for t in originals]
    sections={s['scope']:s for s in model['occ_sections']}
    occ={k:{r['class_name']:r for r in s['classes']} for k,s in sections.items()}
    def item(row,key):
        if not row:return cell()
        return cell(row.get(key),state='no_gt' if row['seen']==0 else 'measured',unit='percent_number')
    rows=[]
    for c in spec['occ_classes']:
        a=occ.get('EVAL',{}).get(c['key']);b=occ.get('EVAL_PILLAR',{}).get(c['key'])
        rows.append([cell(c['label']),*[item(a,k) for k in ('iou','precision','recall')],item(b,'iou')])
    d=metric(model['all_metrics'],'drivable.iou');d['unit']='percent_number'
    rows.append([cell(labels[0][-2][0]),d,*[cell(state='not_applicable') for _ in range(3)]])
    a=sections.get('EVAL',{}).get('non_empty');b=sections.get('EVAL_PILLAR',{}).get('non_empty')
    rows.append([cell(labels[0][-1][0]),*[item(a,k) for k in ('iou','precision','recall')],item(b,'iou')])
    result=[{'id':'occ','headers':labels[0][0],'rows':rows}]
    rows=[]
    for i,t in enumerate(spec['vision_tasks']):
        rows.append([cell(labels[1][i+1][0]),*[metric(model['vision_metrics'],n,task=t['key']) for n in ('vision.map','vision.accuracy','vision.macro_recall','vision.major_classes')]])
    rows.append([cell(labels[1][-1][0]),metric(model['vision_metrics'],'vision.avg_map'),*[cell(state='not_applicable') for _ in range(3)]])
    result.append({'id':'vision_summary','headers':labels[1][0],'rows':rows});rows=[]
    for i,t in enumerate(spec['vision_major_classes']):
        rows.append([cell(labels[2][i+1][0]),cell(labels[2][i+1][1]),*[metric(model['vision_metrics'],n,task=t['task'],class_id=t['class_id']) for n in ('vision.class.support','vision.class.support_percent','vision.class.recall','vision.class.ap')]])
    result.append({'id':'vision_classes','headers':labels[2][0],'rows':rows})
    return result


def render(reference_xml, tables, *, model_version, vision_samples):
    """Retain original model blocks, replacing only tables and sample total.

    The caller supplies the three already validated model tables. There is no
    compile input, execution, publication, extra metric or generated explanation.
    """
    source = ET.fromstring('<root>' + reference_xml + '</root>')
    nodes = list(source)
    stop = next(i for i, n in enumerate(nodes)
                if n.tag == 'h1' and ''.join(n.itertext()) == '板端产物与fp16')
    model = [deepcopy(n) for n in nodes[:stop] if n.tag != 'title']
    originals = [n for n in model if n.tag == 'table']
    if len(originals) != 3 or [t['id'] for t in tables] != ['occ', 'vision_summary', 'vision_classes']:
        raise ValueError('Original three model tables required')
    if not isinstance(vision_samples, int) or vision_samples < 0:
        raise ValueError('Verified visual sample total required')
    out = []
    index = 0
    for node in model:
        text = ''.join(node.itertext())
        # This original block compares two training versions; single-run reports
        # remove cross-version content rather than inventing replacement prose.
        if node.tag == 'blockquote' and 'v3.0.4' in text and 'v3.1.0' in text:
            continue
        if node.tag == 'table':
            item = tables[index]
            original_rows = [[''.join(c.itertext()) for c in row] for row in node.iter('tr')]
            expected_labels = [row[:2] if index == 2 else row[:1]
                               for row in original_rows[(2 if index == 0 else 1):]]
            actual_labels = [[display(c) for c in row[:(2 if index == 2 else 1)]]
                             for row in item['rows']]
            if original_rows[0] != item['headers'] or expected_labels != actual_labels:
                raise ValueError('Original table headers or row labels differ')
            node = ET.fromstring(table(item['headers'], [[display(c) for c in row] for row in item['rows']]))
            index += 1
        else:
            if '模型版本：' in text and model_version not in text:
                raise ValueError('Model version differs from the supplied template identity')
            for child in node.iter():
                if child.text:
                    child.text = re.sub(r'(Total samples\s*=\s*)\d+',
                                        lambda m: m.group(1) + str(vision_samples), child.text)
        for child in node.iter():
            for attr in ('id', 'seq-marker'):
                child.attrib.pop(attr, None)
        out.append(node)
    if index != 3:
        raise ValueError('Incomplete model template projection')
    return ''.join(ET.tostring(n, encoding='unicode') for n in out)
