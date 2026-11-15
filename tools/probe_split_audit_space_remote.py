import json,pathlib
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027')
print(json.dumps({'base_exists':base.is_dir(),'source_exists':(base/'source').is_dir(),'main_exists':(base/'weights/epoch_44.pth').is_file(),'prior_trajectory_exists':(base/'heads-20261027/trajectory-01/launch.json').exists()}))
