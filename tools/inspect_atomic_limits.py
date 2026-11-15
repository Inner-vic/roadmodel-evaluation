import json
from pathlib import Path

root=Path('evidence/reports/compile-atomic-v3-20261027')
report=json.loads((root/'report.json').read_text(encoding='utf-8'))
finite=report['source_normalized_result']['provenance']['input_validation']['finite_outputs']
print(json.dumps({'finite_keys':list(finite),'checks_type':type(finite['checks']).__name__,
                  'checks_first':finite['checks'][:1] if isinstance(finite['checks'],list) else finite['checks'],
                  'frames_first':finite.get('frames',[])[:1] if isinstance(finite.get('frames'),list) else finite.get('frames'),
                  'xml_sizes':{p.name:p.stat().st_size for p in (root/'report.sections').glob('*.xml')}},ensure_ascii=True)[:4000])
