"""Template-bound creation using the official SDK's authenticated REST client."""
import json
import os
from pathlib import Path
import subprocess
import time

from .common import save_json


class NotebookApi:
    def __init__(self,sdk_python,request_file,root):
        self.python=sdk_python;self.request_file=str(Path(request_file).resolve())
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.worker=Path(__file__).resolve().parents[1]/'tools/notebook_sdk_worker.py'

    def invoke(self,action,body=None,journal=None):
        command=[self.python,'-X','utf8','-B',str(self.worker),action,'--request',self.request_file]
        if body:command+=['--body',str(body)]
        if journal:command+=['--journal',str(journal)]
        try:
            result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=75,
                **({'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}))
        except subprocess.TimeoutExpired:
            return {'state':'UNKNOWN','reason':'SDK worker timed out; reconcile original name only'}
        if result.returncode:raise ValueError('Official Notebook SDK worker failed; raw diagnostics suppressed')
        value=json.loads(result.stdout)
        if not isinstance(value,dict):raise ValueError('Official Notebook response shape differs')
        return value

    def prepare(self,detail,asset):
        observed=self.invoke('inspect')
        if observed.get('state')!='OBSERVED' or observed.get('owner_verified') is not True:
            raise ValueError('Verify the current account through the official SDK first')
        req=detail['request'];template=req['template'];services=observed['notebook_config']['accesses'];access=[]
        for item in services['required']+services['optional']:
            if item['name'] not in ('jupyter-lab','ssh'):
                if item in services['required']:raise ValueError('Template contains an unreviewed required service')
                continue
            item={k:v for k,v in item.items() if k in ('name','targetPort','protocol','address')}
            if item['name']=='ssh':item['pubkeyName']=req['ssh']['credential_name']
            access.append(item)
        path='/mnt/datasets/'+asset['name']+'/'+asset['version']
        body={'metadata':{'name':detail['space_name'],'namespace':template['namespace'],
            'labels':{'lpai.org/kind':'notebook','lpai.org/visible':'personal'},
            'annotations':{'lpai.org/desc':'RoadModel专属评测空间；按本次输入挂载数据；不复制数据集。',
                           'lpai.org/gems-project':template['project']}},
            'spec':{'image':template['image'],'clusterName':template['cluster'],
                'sshSecretName':req['ssh']['credential_name'],'resources':template['resources'],
                'autoCulling':{'enable':'true','windowMinutes':'240'},'env':{},'accesses':access,
                'dataVols':[{'type':'dataset','id':asset['dataset_uid'],'name':asset['name'],'version':asset['version'],
                    'fsType':'ofs','mountPath':path,'envKey':'LPAI_INPUT_DATASET_0','envValue':path}]}}
        from .common import load_json
        target=self.root/'create-body.json'
        if target.exists():
            if load_json(target)!=body:raise ValueError('Original Notebook creation body differs')
        else:save_json(target,body)
        return target

    def observe(self,detail,body):
        value=self.invoke('read',body)
        if value.get('metadata',{}).get('uid'):
            return {'exists':True,'value':value}
        if value.get('state')=='HTTP_ERROR' and value.get('http_status')==404:
            return {'exists':False,'receipt':{'source':'LPAI_NOTEBOOK_API','real_api_call_completed':True,
                'space_name':detail['space_name'],'lpai_login':detail['request']['account']['lpai_login'],
                'owner':detail['request']['account']['lpai_account'],
                'namespace':detail['request']['template']['namespace'],'cluster':detail['request']['template']['cluster'],
                'http_status':404,'exists':False,'observed_at':time.time()}}
        return {'exists':None,'value':value}

    def create(self,body):
        journal=self.root/'create-request.json'
        if journal.exists():return {'state':'RECONCILE_ONLY','new_requests':0}
        return self.invoke('create',body,journal)
