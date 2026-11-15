import json
from pathlib import Path

root = Path('inspection/gt-source-524e07972382')
for name, spans in [
    ('config/2-1-0.py', [(285,325), (410,440)]),
    ('tools/dataset_process/Occ&dataLink/occlabel_26labels_v3_1.py', [(380,410), (1190,1240)]),
    ('misc/eval/metric_util.py', [(180,300)]),
]:
    lines = (root / name).read_text(encoding='utf-8').splitlines()
    print(name)
    for start,end in spans:
        for i in range(start-1,min(end,len(lines))):
            print(str(i+1)+': '+lines[i].encode('ascii','backslashreplace').decode())
