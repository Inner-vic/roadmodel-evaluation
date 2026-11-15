"""Hash-bound model environment migration; dataset mounts never enter the bundle."""
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

from .common import digest, load_json, save_json
from .planning import ssh_command


def allowed_path(name):
    p = PurePosixPath(name)
    return (not p.is_absolute() and '..' not in p.parts and
            (name == 'source-manifest.json' or p.parts[0] in ('source', 'weights')))


def verify_bundle(path):
    with tarfile.open(path, 'r:gz') as tar:
        manifest = json.load(tar.extractfile('environment-manifest.json'))
        names = set()
        import hashlib
        for member in tar:
            if member.name == 'environment-manifest.json':
                continue
            if not member.isfile() or not allowed_path(member.name) or member.name in names:
                raise ValueError('环境包包含不允许的文件、链接或重复成员')
            names.add(member.name)
            if hashlib.sha256(tar.extractfile(member).read()).hexdigest() != manifest['files'].get(member.name):
                raise ValueError('环境包文件哈希不一致')
        if names != set(manifest['files']):
            raise ValueError('环境包文件清单不完整')
    return manifest


def capture(profile, space, output):
    """Explicitly authorized LPAI -> WS snapshot of model files only."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    archive, proof = root/'environment.tar.gz', root/'bundle.json'
    if proof.exists():
        value = load_json(proof)
        if digest(archive) != value['sha256']:
            raise ValueError('原环境归档发生变化')
        return value
    partial = root/'environment.partial'
    if partial.exists() or archive.exists():
        raise ValueError('已有环境传输结果，需先核验，不能重复覆盖')
    base = str(PurePosixPath(profile['source_root']).parent)
    collector = '''import pathlib,json,sys,hashlib,tarfile,io,importlib.metadata,sysconfig
base=pathlib.Path(sys.argv[1]).resolve();profile=json.loads(sys.argv[2]);source=base/'source'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((base/'source-manifest.json').read_text());assert manifest['commit']==profile['source_commit']
entries=manifest['files'];entries=({r['path']:r['sha256'] for r in entries} if isinstance(entries,list) else entries)
files={}
for name,expected in entries.items():
 p=(source/name).resolve();p.relative_to(source);assert p.is_file() and sha(p)==expected;files['source/'+name]=expected
for p in source.rglob('*.so'):
 if 'build' in p.relative_to(source).parts:continue
 assert not p.is_symlink();p=p.resolve();p.relative_to(source);files[p.relative_to(base).as_posix()]=sha(p)
for field in ('main_checkpoint','classifier_checkpoint'):
 p=pathlib.Path(profile[field]['path']).resolve();p.relative_to(base);assert sha(p)==profile[field]['sha256'];files[p.relative_to(base).as_posix()]=sha(p)
files['source-manifest.json']=sha(base/'source-manifest.json')
versions={}
for package in ('torch','numpy','mmengine','mmcv','mmsegmentation','mmdet','scipy','einops'):
 try:versions[package]=importlib.metadata.version(package)
 except importlib.metadata.PackageNotFoundError:pass
body=json.dumps({'schema':'roadmodel.environment/v1','source_commit':profile['source_commit'],'files':files,'python_abi':sysconfig.get_config_var('SOABI'),'packages':versions,'dataset_files_included':False}).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 info=tarfile.TarInfo('environment-manifest.json');info.size=len(body);tar.addfile(info,io.BytesIO(body))
 for name in files:
  p=base/name;assert p.is_file() and not p.is_symlink();tar.add(p,arcname=name,recursive=False)
'''
    options = {k: space[k] for k in ('user', 'port', 'identity_file')}
    with partial.open('xb') as handle:
        result = subprocess.run(ssh_command(space['host'], ['python3', '-c', collector, base, json.dumps(profile)], options=options),
                                stdout=handle, stderr=subprocess.PIPE, timeout=300)
    if result.returncode:
        raise ValueError('环境归档失败；保留partial，不自动重传：'+result.stderr.decode(errors='replace')[-300:])
    manifest = verify_bundle(partial)
    partial.replace(archive)
    value = {'state': 'CAPTURED', 'path': str(archive), 'sha256': digest(archive),
             'bytes': archive.stat().st_size, 'environment_root': base,
             'source_commit': profile['source_commit'], 'files': len(manifest['files']),
             'dataset_files_included': False, 'python_abi': manifest['python_abi'], 'packages': manifest['packages']}
    save_json(proof, value)
    return value


def ensure(bundle, space):
    from .lpai_model_transport import call
    archive = Path(bundle['path'])
    if digest(archive) != bundle['sha256']:
        raise ValueError('模型环境归档哈希不同')
    manifest = verify_bundle(archive)
    # Source and weights must not overwrite different existing files; native
    # operators must match the target interpreter and torch ABI.
    probe = '''import pathlib,json,sys,hashlib,sysconfig,importlib.metadata
root=pathlib.Path(sys.argv[1]);m=json.load(sys.stdin);missing=[]
assert sysconfig.get_config_var('SOABI')==m['python_abi'],'Python ABI differs'
for name,expected in m['packages'].items():assert importlib.metadata.version(name)==expected,'Package version differs: '+name
for name,expected in m['files'].items():
 p=(root/name).resolve();p.relative_to(root)
 if not p.is_file():missing.append(name)
 elif hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('Existing environment differs: '+name)
print(json.dumps({'missing':len(missing)}))'''
    checked = json.loads(call(space, ['python3', '-c', probe, bundle['environment_root']], json.dumps(manifest).encode()))
    if not checked['missing']:
        return {'state': 'READY', 'uploaded_bytes': 0, 'environment_sha256': bundle['sha256']}
    target = '/lpai/roadmodel-evaluation/bootstrap/'+bundle['sha256']+'.tar.gz'
    receiver = '''import pathlib,sys,hashlib,os
p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True)
if p.exists():
 assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2];sys.stdin.buffer.read();print('REUSED');sys.exit()
tmp=p.with_suffix('.partial');h=hashlib.sha256()
with tmp.open('xb') as f:
 for b in iter(lambda:sys.stdin.buffer.read(1024*1024),b''):f.write(b);h.update(b)
assert h.hexdigest()==sys.argv[2];os.replace(tmp,p)
'''
    options = {k: space[k] for k in ('user', 'port', 'identity_file')}
    with archive.open('rb') as handle:
        result = subprocess.run(ssh_command(space['host'], ['python3', '-c', receiver, target, bundle['sha256']], options=options),
                                stdin=handle, capture_output=True, timeout=300)
    if result.returncode:
        raise ValueError('环境上传待核验，保留原partial，不重复提交')
    installer = '''import pathlib,sys,tarfile,json,hashlib
root=pathlib.Path(sys.argv[1]);archive=pathlib.Path(sys.argv[2]);assert hashlib.sha256(archive.read_bytes()).hexdigest()==sys.argv[3]
with tarfile.open(archive,'r:gz') as tar:
 m=json.load(tar.extractfile('environment-manifest.json'));seen=set()
 for member in tar:
  if member.name=='environment-manifest.json':continue
  assert member.isfile() and member.name in m['files'] and member.name not in seen;seen.add(member.name)
  p=(root/member.name).resolve();p.relative_to(root);assert pathlib.PurePosixPath(member.name).parts[0] in ('source','weights') or member.name=='source-manifest.json'
  body=tar.extractfile(member).read();assert hashlib.sha256(body).hexdigest()==m['files'][member.name];p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==body
  else:
   with p.open('xb') as f:f.write(body)
 assert seen==set(m['files'])
print(json.dumps({'state':'READY','files':len(seen)}))'''
    installed = json.loads(call(space, ['python3', '-c', installer, bundle['environment_root'], target, bundle['sha256']], timeout=180))
    return {**installed, 'uploaded_bytes': archive.stat().st_size, 'environment_sha256': bundle['sha256']}
