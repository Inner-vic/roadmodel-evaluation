import argparse,shutil
import deploy_initialized_space_20261109 as deploy
deploy.SOURCE='/mnt/data/bench-eval/service/roadmodel/evaluation-content-parser-20261109-r9'
deploy.RELEASE='/mnt/data/bench-eval/service/roadmodel/evaluation-card-diagnostics-20261109-r10'
deploy.STAGE='/mnt/data/bench-eval/roadmodel/deployment/card-diagnostics-20261109-r10'
original=deploy.OUT
deploy.OUT=deploy.ROOT/'evidence/card-acceptance-full-20261109/release'
deploy.BACKUP=deploy.ROOT/'.runtime/evaluation-desktop-before-card-diagnostics-r10'
deploy.SELECTION={'scene_count':1,'frames_per_scene':2,'frame_offset':4}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','deploy'));args=p.parse_args()
 if args.action=='prepare':
  (deploy.OUT/'plan').mkdir(parents=True,exist_ok=True)
  target=deploy.OUT/'plan/initialization.json'
  if not target.exists():shutil.copy2(original/'plan/initialization.json',target)
 deploy.main(args.action)
