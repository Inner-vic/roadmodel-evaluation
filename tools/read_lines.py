import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('start',type=int);p.add_argument('end',type=int)
a=p.parse_args()
for i,line in enumerate(Path(a.path).read_text(encoding='utf-8').splitlines(),1):
 if a.start<=i<=a.end:print(f'{i}: {line}')
