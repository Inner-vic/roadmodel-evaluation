"""Narrow Notebook API transport through the installed official SDK.

Uses standard LpaiConf authentication in this process only. Never returns
credentials, arbitrary headers, access URLs, or unfiltered exception text.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import sys


def safe_error(exc):
    value=str(exc)
    status=re.search(r"Invalid status code '([0-9]{3})'",value)
    delimiter=next((x for x in (' with response: ',' with error payload: ') if x in value),None)
    body=value.split(delimiter,1)[-1] if delimiter else ''
    # Only scalar error text is retained. Drop echoed credentials/requests.
    try:parsed=json.loads(body)
    except ValueError:parsed={}
    def clean(text):
        text=str(text)
        if re.search(r'(?i)authorization|lpaiJwt|bearer|access.token|refresh.token|password|[\"\x27]token[\"\x27]\s*:|eyJ[A-Za-z0-9_-]{12}',text):
            return '[redacted sensitive diagnostic]'
        return text[:1600]
    fields={k:clean(parsed[k]) for k in ('code','message','msg','error','detail','reason')
            if isinstance(parsed.get(k),(str,int,float))}
    if isinstance(parsed.get('error'),dict):
        fields['error']={k:clean(parsed['error'][k]) for k in ('code','message','msg','reason')
                         if isinstance(parsed['error'].get(k),(str,int,float))}
    return {'state':'HTTP_ERROR','http_status':int(status[1]) if status and delimiter==' with response: ' else None,
            'business_code':int(status[1]) if status and delimiter==' with error payload: ' else None,
            'error_type':type(exc).__name__,'response':fields,
            'response_text':clean(body) if not fields else None,
            'response_kind':'http' if delimiter==' with response: ' else 'business' if delimiter else 'unknown',
            'raw_response_saved':False}


def unbox(raw):
    while isinstance(raw,dict) and set(raw)&{'data','result'} and not set(raw)&{'metadata','items'}:
        key='result' if raw.get('result') is not None else 'data'
        if raw.get(key) is None:break
        raw=raw[key]
    return raw


def summarize(raw):
    raw=unbox(raw);meta=raw['metadata'];spec=raw['spec']
    annotations=meta.get('annotations',{})
    return {'metadata':{k:meta.get(k) for k in ('name','uid','namespace')},
        'owner':annotations.get('lpai.org/owner'),
        'base_image':annotations.get('lpai.org/baseImage',annotations.get('lpai.org/base-image')),
        'project':annotations.get('lpai.org/gems-project'),
        'labels':{k:v for k,v in meta.get('labels',{}).items() if k in ('lpai.org/visible','lpai.org/resourceQueue')},
        'spec':{k:spec.get(k) for k in ('image','clusterName','resources','autoCulling','dataVols','sshSecretName')},
        'accesses':[{k:x.get(k) for k in ('name','targetPort','protocol','pubkeyName')} for x in spec.get('accesses',[])],
        'status':{k:raw.get('status',{}).get(k) for k in ('phase','phaseLan','message')}}


def validate_body(req,body):
    from urllib.parse import urlsplit,parse_qs
    meta=body['metadata'];spec=body['spec'];template=req['template']
    assert set(body)=={'metadata','spec'} and meta['namespace']==template['namespace']
    assert set(meta)=={'name','namespace','labels','annotations'}
    assert set(spec)=={'image','clusterName','sshSecretName','resources','autoCulling','env','dataVols','accesses'}
    assert meta['labels']=={'lpai.org/kind':'notebook','lpai.org/visible':'personal'}
    assert set(meta['annotations'])=={'lpai.org/desc','lpai.org/gems-project'} and meta['annotations']['lpai.org/gems-project']==template['project']
    assert spec['image']==template['image'] and spec['clusterName']==template['cluster']
    assert spec['resources']==template['resources'] and spec['sshSecretName']==req['ssh']['credential_name']
    assert spec['autoCulling']=={'enable':'true','windowMinutes':'240'} and spec['env']=={}
    ref=urlsplit(req['dataset_ref']);version=parse_qs(ref.query)['version'][0]
    assert ref.hostname=='lpai.lixiang.com' and len(spec['dataVols'])==1
    vol=spec['dataVols'][0]
    assert set(vol)=={'type','id','name','version','fsType','mountPath','envKey','envValue'}
    assert vol['type']=='dataset' and vol['id']==ref.path.rsplit('/',1)[-1] and vol['version']==version
    assert vol['fsType']=='ofs' and vol['mountPath']==vol['envValue']=='/mnt/datasets/'+vol['name']+'/'+version
    assert vol['envKey']=='LPAI_INPUT_DATASET_0'
    assert sorted(x['name'] for x in spec['accesses'])==['jupyter-lab','ssh']
    assert next(x for x in spec['accesses'] if x['name']=='ssh')['pubkeyName']==spec['sshSecretName']


def run(action,request_file,body_file=None,journal=None):
    req=json.loads(Path(request_file).read_text(encoding='utf-8-sig'))
    ns=req['template']['namespace'];owner=req['account']['lpai_account']
    assert re.fullmatch(r'[a-z0-9-]+',ns)
    # The SDK loads its existing credential. The application never reads or
    # copies the credential value or writes a new SDK configuration.
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        from lpai.client.rest_client import RestClient
        from lpai.conf.lipai_conf import LpaiConf
        conf=LpaiConf()
        if conf.get_env()!='prod' or conf.get_namespace()!=ns:raise ValueError('SDK production namespace differs')
        client=RestClient('https://lpai.lixiang.com',{'Content-Type':'application/json'},conf)
    endpoint='/lpai/api/kubeflow/namespaces/'+ns+'/notebooks'
    def get(path):
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                return unbox(client.get(path,timeout=30))
        except Exception as exc:return safe_error(exc)
    rows=get(endpoint)
    if isinstance(rows,dict) and rows.get('state')=='HTTP_ERROR':return rows
    while isinstance(rows,dict):
        rows=rows[next(k for k in ('items','result','data') if k in rows)]
    if not isinstance(rows,list):raise ValueError('Official Notebook list shape differs')
    own=[s for s in rows if s.get('metadata',{}).get('annotations',{}).get('lpai.org/owner')==owner]
    if not own:raise ValueError('SDK identity has no verified spaces belonging to this account')
    if action=='inspect':
        config=get('/lpai/api/kubeflow/notebook-config')
        return {'state':'OBSERVED','sdk_version':__import__('importlib.metadata',fromlist=['version']).version('lpai'),
            'owner_verified':True,'spaces':[summarize(s) for s in own],
            'notebook_config':config,'platform_writes':0}
    if action=='nodes':
        labels=','.join(k+'='+v for k,v in req['template']['resources']['affinity'].items() if k!='kubernetes.io/hostname')
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                result=unbox(client.get('/lpai/api/kubeflow/nodes',data={'cluster':req['template']['cluster'],'labels':labels},timeout=30))
        except Exception as exc:return safe_error(exc)
        nodes=result.get('node',[]) if isinstance(result,dict) else []
        return {'state':'OBSERVED','nodes':[{'name':n['metadata']['name'],'labels':{
            k:v for k,v in n['metadata'].get('labels',{}).items() if k.startswith('lpai.org/share-')},
            'conditions':n.get('status',{}).get('conditions',[])} for n in nodes],
            'selector':labels,'platform_writes':0}
    body=json.loads(Path(body_file).read_text(encoding='utf-8-sig'))
    name=body['metadata']['name']
    if not re.fullmatch(r'[a-z][a-z0-9-]{0,39}',name):raise ValueError('Invalid exact Notebook name')
    exact=get(endpoint+'/'+name)
    if action=='read':return exact if exact.get('state')=='HTTP_ERROR' else summarize(exact)
    if action!='create':raise ValueError('Unsupported Notebook operation')
    if exact.get('state')!='HTTP_ERROR' or exact.get('http_status')!=404:
        return {'state':'RECONCILE_ONLY','reason':'Exact target is not confirmed HTTP404','platform_writes':0}
    if sum(s['spec']['clusterName']==req['template']['cluster'] for s in own)>=5:
        return {'state':'BLOCKED_CAPACITY','platform_writes':0}
    validate_body(req,body)
    # Exclusive journal is created BEFORE the single POST. No implicit retry.
    attempt={'state':'POST_REQUESTED','name':name,'body_sha256':hashlib.sha256(Path(body_file).read_bytes()).hexdigest()}
    with Path(journal).open('x',encoding='utf-8') as f:json.dump(attempt,f)
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            result=client.post(endpoint,body)
        result={'state':'POST_RETURNED','response':summarize(result) if isinstance(unbox(result),dict) and 'metadata' in unbox(result) else {'accepted':True}}
    except Exception as exc:result=safe_error(exc)
    result.update(platform_writes=1,name=name)
    Path(journal).write_text(json.dumps({**attempt,'result':result},indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('inspect','nodes','read','create'));p.add_argument('--request',required=True)
    p.add_argument('--body');p.add_argument('--journal');args=p.parse_args()
    try:value=run(args.action,args.request,args.body,args.journal)
    except Exception as exc:value={'state':'TRANSPORT_BLOCKED','error_type':type(exc).__name__}
    print(json.dumps(value,ensure_ascii=True))
