"""Deploy the tested explicit model-field alias before a fresh two-frame card."""
import argparse
import shutil
import deploy_initialized_space_20261109 as deploy

deploy.SOURCE=deploy.RELEASE
deploy.RELEASE='/mnt/data/bench-eval/service/roadmodel/evaluation-model-field-20261109-r8'
deploy.STAGE='/mnt/data/bench-eval/roadmodel/deployment/model-field-20261109-r8'
original=deploy.OUT
deploy.OUT=original/'parser-fix-release'
deploy.BACKUP=deploy.ROOT/'.runtime/evaluation-desktop-before-model-field-r8'
deploy.SELECTION={'scene_count':1,'frames_per_scene':2,'frame_offset':2}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','deploy'));args=p.parse_args()
    if args.action=='prepare':
        (deploy.OUT/'plan').mkdir(parents=True,exist_ok=True)
        target=deploy.OUT/'plan/initialization.json'
        if not target.exists():shutil.copy2(original/'plan/initialization.json',target)
    deploy.main(args.action)
