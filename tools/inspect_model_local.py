from pathlib import Path
import json,re,subprocess
for p in [Path('AGENTS.md'),Path('inspection/AGENTS.md'),Path('tools/AGENTS.md'),Path('pcm_eval/AGENTS.md')]:
 if p.exists(): print(str(p)+'\n'+p.read_text(encoding='utf-8'))
root=Path('inspection/release-model-source/9e16e29351d17a726ce6beca6f821de79b7ade40')
for name,pattern in [('eval.py',r'def main|distributed =|load_state_dict|init_dist|init_process|data_link|build_segmentor'),('dataset/dataset.py',r'class SmartChassisDatasetV7|class SmartChassisDatasetV[456]|def get_data_info|_resolve_linked_file_path'),('config/3-0-0-temporal-dit.py',r'os.environ|os.getenv|DATASET|DATA_LINK|NUM_FRAME|DATA_PATH|HEAD_MODE|TRAIN_MODE'),('config/2-1-0.py',r'os.environ|os.getenv|DATASET|DATA_LINK|NUM_FRAME|DATA_PATH|CLASSIFIER')]:
 print('\nFILE',name)
 for i,line in enumerate((root/name).read_text(encoding='utf-8').splitlines(),1):
  if re.search(pattern,line):print(f'{i}: {line}')
link=json.loads(Path('evidence/model-evaluation/model-reference-20261027/two-frame-data-link.json').read_text())
print('DATA_LINK',json.dumps(link,ensure_ascii=False)[:10000])
print('TREE_SIZES',subprocess.run(['git','-C','inspection/roadmodel-dev-v41','ls-tree','-r','-l','9e16e29351d17a726ce6beca6f821de79b7ade40'],capture_output=True,text=True).stdout[:1000])
