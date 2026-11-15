"""Create isolated, explicitly test-only ambiguous delivery; never alter originals."""
import json
from pathlib import Path
from deploy_environment_flow import remote
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/card-acceptance-full-20261109'
CODE='''import pathlib,json,sys,subprocess,onnx
from onnx import helper,TensorProto
root=pathlib.Path('/mnt/data/bench-eval/service/roadmodel/evaluation-content-parser-20261109-r9')
cfg=json.loads((root/'configs/evaluation.json').read_text());lark=cfg['workstation_lark']
out=pathlib.Path('/mnt/data/bench-eval/roadmodel/validation/card-acceptance-full-20261109');out.mkdir(parents=True,exist_ok=True)
receipt=out/'ambiguity-fixture.json'
if receipt.exists():print(receipt.read_text());sys.exit()
def cli(args,body=None):
 p=subprocess.run([lark['executable'],'--profile',lark['profile'],*args,'--as','user'],input=body,capture_output=True,text=True,cwd=out,timeout=90)
 try:v=json.loads(p.stdout or p.stderr)
 except ValueError:raise ValueError('CLI output: '+(p.stderr[:900] if not any(x in p.stderr.lower() for x in ('bearer','lpaiJwt','access_token','appsecret')) else 'redacted'))
 assert p.returncode==0 and v.get('ok') is True,{'exit_code':p.returncode,'error':v.get('error',{})}
 return v['data']
folder_file=out/'test-folder.json'
if folder_file.exists():folder=json.loads(folder_file.read_text())
else:
 folder=cli(['drive','files','create_folder','--data','-'],json.dumps({'name':'RoadModel解析歧义测试-20261109-仅验证不执行','folder_token':''}));folder_file.write_text(json.dumps(folder))
token=folder.get('token') or folder.get('folder_token');assert token
files=[]
for letter in ('A','B'):
 name='TEST_ONLY_ambiguity_'+letter+'_v3.2.2.onnx';path=out/name
 model=helper.make_model(helper.make_graph([helper.make_node('Identity',['x'],['y'])],'ambiguity_test_'+letter,
  [helper.make_tensor_value_info('x',TensorProto.FLOAT,[1])],[helper.make_tensor_value_info('y',TensorProto.FLOAT,[1])]),opset_imports=[helper.make_opsetid('',13)])
 onnx.checker.check_model(model);onnx.save(model,path)
 done=out/(letter+'-upload.json')
 if done.exists():uploaded=json.loads(done.read_text())
 else:
  uploaded=cli(['drive','+upload','--file','./'+name,'--folder-token',token]);done.write_text(json.dumps(uploaded))
 files.append({'name':name,'bytes':path.stat().st_size})
v={'url':'https://li.feishu.cn/drive/folder/'+token,'files':files,'test_only':True,'intended_inference_frames':0,'original_delivery_modified':False}
receipt.write_text(json.dumps(v));print(json.dumps(v))
'''
if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True);v=remote(CODE,timeout=160)
 (OUT/'ambiguity-fixture.json').write_text(json.dumps(v,indent=2),encoding='utf-8');print(json.dumps(v))
