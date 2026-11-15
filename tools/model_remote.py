"""Run a scoped remote Python file through the existing model-space connection."""
import argparse
from pathlib import Path
from model_space_transport import run

p=argparse.ArgumentParser()
p.add_argument('script', type=Path)
p.add_argument('--output', type=Path)
p.add_argument('--timeout', type=int, default=120)
a=p.parse_args()
data=run(['python3','-'],a.script.read_bytes(),timeout=a.timeout)
if a.output:
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_bytes(data)
print(data.decode(errors='replace'))
