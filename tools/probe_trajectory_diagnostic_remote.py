import json,pathlib,importlib.util
base=pathlib.Path('/lpai/roadmodel-single-eval/model-v3.2.2-smoke-20261027')
out={'base_exists':base.is_dir(),'torch_available':importlib.util.find_spec('torch') is not None,
     'existing_trajectory_run':(base/'heads-20261027/trajectory-01').is_dir(),
     'diagnostic_exists':(base/'heads-20261027/trajectory-diagnostic-01').exists()}
if out['torch_available']:
    import torch
    out.update(torch=torch.__version__,cuda=torch.cuda.is_available())
    if out['cuda']:out['gpu1_free_total']=list(torch.cuda.mem_get_info(1))
print(json.dumps(out))
