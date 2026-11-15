"""Resolve a submitted LPAI asset and verify its mount, without choosing defaults.

Platform mutations deliberately stay outside this module. An unmounted asset is
an explicit preparation state, never permission to use a workstation cache.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import parse_qs, urlsplit


def dataset_reference(value):
    """Only an explicit, versioned official dataset link identifies an asset."""
    u = urlsplit(value)
    if (u.scheme != 'https' or u.hostname != 'lpai.lixiang.com'
            or u.username or u.password or u.port not in (None, 443) or u.fragment):
        raise ValueError('请提供带版本的LPAI数据集链接')
    match = re.fullmatch(r'/lpaiweb/xspacex_([A-Za-z0-9_-]+)/subApp/dataset/detail/([A-Za-z0-9_-]+)/?', u.path)
    query = parse_qs(u.query, keep_blank_values=True)
    versions = query.get('version', [])
    if not match or len(versions) != 1 or not re.fullmatch(r'[A-Za-z0-9_.-]+', versions[0]):
        raise ValueError('数据集链接必须明确数据集和唯一版本；不会使用最新版或空间默认版本')
    ns, identifier = match.groups()
    return {'source_ref': value, 'namespace': ns, 'identifier': identifier,
            'identifier_kind': 'uid' if re.fullmatch(r'[a-fA-F0-9]{32}', identifier) else 'name',
            'version': versions[0]}


def resolve_dataset(value, query):
    """query is the existing authenticated official CLI adapter, not HTTP."""
    ref = dataset_reference(value)
    raw = query(['assets', 'datasets', 'version', '--namespace', ref['namespace'],
                 '--' + ref['identifier_kind'], ref['identifier'], '--version', ref['version']])
    meta = raw['metadata']
    namespace = [r['uid'] for r in meta.get('ownerReferences', []) if r.get('kind') == 'namespace']
    if namespace != [ref['namespace']] or meta.get('name') != ref['version']:
        raise ValueError('平台返回的数据集命名空间或版本与本次输入不一致')
    if ref['identifier_kind'] == 'uid' and meta.get('pid') != ref['identifier']:
        raise ValueError('平台返回的数据集UID与本次输入不一致')
    if ref['identifier_kind'] == 'name' and meta.get('pName') != ref['identifier']:
        raise ValueError('平台返回的数据集名称与本次输入不一致')
    if meta.get('labels', {}).get('status') != 'AVAILABLE':
        raise ValueError('本次数据集版本当前不可用')
    source = raw.get('spec', {}).get('dir', {}).get('path')
    if not source or not meta.get('uid') or not meta.get('pid') or not meta.get('pName'):
        raise ValueError('平台没有返回完整的数据集来源标识')
    asset = {k: ref[k] for k in ('source_ref', 'namespace', 'version')}
    asset.update(dataset_uid=meta['pid'], version_uid=meta['uid'], name=meta['pName'], source_uri=source)
    # Hash only the identity; timestamps and access counters are not data versions.
    asset['binding_sha256'] = hashlib.sha256(json.dumps(asset, sort_keys=True).encode()).hexdigest()
    return asset


def resolve_dataset_input(value,query,space):
    """A path identifies its actual platform volume, never a catalog/cache alias."""
    if value.startswith('https://'):return resolve_dataset(value,query)
    if value.startswith('工作站:'):
        raise ValueError('模型评测在LPAI执行，请提供这份数据的LPAI版本链接或开发空间挂载路径，不能直接读取工作站目录')
    path=value.removeprefix('开发空间:');p=PurePosixPath(path)
    if not p.is_absolute() or '..' in p.parts:raise ValueError('开发空间路径必须是完整绝对路径')
    matches=[]
    for v in space.get('spec',{}).get('dataVols',[]):
        if v.get('type')!='dataset' or not v.get('mountPath'):continue
        root=PurePosixPath(v['mountPath'])
        if p==root or root in p.parents:matches.append(v)
    if len(matches)!=1:raise ValueError('该路径没有唯一的数据集版本挂载关联，请提供对应LPAI版本链接，不能按目录名猜版本')
    v=matches[0];root=PurePosixPath(v['mountPath']);rel=p.relative_to(root)
    if p!=root and p.suffix!='.json':raise ValueError('请提供数据集挂载根目录或明确的索引JSON路径，不能把场景子目录默认扩展为完整数据集')
    ns=space['metadata']['namespace']
    ref=f"https://lpai.lixiang.com/lpaiweb/xspacex_{ns}/subApp/dataset/detail/{v['name']}?version={v['version']}"
    asset=resolve_dataset(ref,query);asset.pop('binding_sha256')
    asset['source_ref']=value
    if p!=root:asset['index_relative_hint']=rel.as_posix()
    asset['binding_sha256']=hashlib.sha256(json.dumps(asset,sort_keys=True).encode()).hexdigest()
    return asset


def mount_plan(asset, space):
    """Read actual devspace spec; never infer a PVC or reuse a preheat path."""
    meta, spec = space.get('metadata', {}), space.get('spec', {})
    uid = meta.get('uid')
    if not uid:
        raise ValueError('缺少开发空间UID，不能绑定运行位置')
    namespaces = [r['uid'] for r in meta.get('ownerReferences', []) if r.get('kind') == 'namespace']
    ns = meta.get('namespace') or (namespaces[0] if len(namespaces) == 1 else None)
    if ns != asset['namespace']:
        raise ValueError('数据集与开发空间不在同一命名空间，需核对实际授权和挂载')
    candidates = [v for v in spec.get('dataVols', []) if v.get('type') == 'dataset'
                  and v.get('name') == asset['name'] and v.get('version') == asset['version']]
    phase = space.get('status', {}).get('phase', 'unknown').lower()
    plan = {'schema': 'roadmodel.dataset-mount/v1', 'asset': asset, 'space_uid': uid,
            'space_name': meta.get('name'), 'space_phase': phase,
            'state': 'MOUNT_REQUIRED', 'evaluation_allowed': False,
            'preserve_existing_mounts': True, 'lifecycle_managed_by': 'agent'}
    if len(candidates) > 1:
        raise ValueError('同一数据集版本有多个挂载，必须明确本次使用的位置')
    if not candidates:
        plan['state'] = 'SPACE_STOP_REQUIRED' if phase in ('ready', 'running', 'pending', 'starting') else 'MOUNT_REQUIRED'
        plan['reason'] = 'Agent核对空间占用后，通过LPAI原生页面停空间、增加本次只读挂载、启动并核验；不依赖用户代操作'
        return plan
    volume = candidates[0]
    root = PurePosixPath(volume.get('mountPath', ''))
    ui_readback = space.get('verification', {}).get('source') == 'LPAI_UI'
    if not root.is_absolute() or '..' in root.parts or str(root) == '/' or (not volume.get('pvcName') and not ui_readback):
        raise ValueError('平台挂载缺少有效路径或卷标识')
    if volume.get('mode') != 'ReadOnly':
        raise ValueError('评测数据必须使用只读挂载')
    plan.update(state='MOUNT_CONFIGURED', mount={k: volume.get(k) for k in
                ('name', 'version', 'mountPath', 'pvcName', 'mode', 'envKey', 'envValue', 'fsType')})
    # Configured is not mounted/readable. A separate in-space probe is required.
    return plan


def probe_mounted_index(plan, index_relative=None, *, mountinfo_path='/proc/self/mountinfo', include_manifest=False):
    """Run INSIDE the target LPAI space; no workstation data/index reads."""
    if plan.get('state') != 'MOUNT_CONFIGURED':
        raise ValueError('本次数据尚未配置挂载')
    if plan.get('space_phase') != 'ready':
        raise ValueError('空间尚未就绪，不能把旧容器的目录探测当成本次挂载成功')
    root = Path(plan['mount']['mountPath'])
    if not root.is_dir() or root.resolve() != root:
        raise ValueError('本次平台挂载目录不存在或指向其他目录')
    # mountinfo escapes whitespace using octal sequences, not URL escapes.
    unescape = lambda s: re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), s)
    rows = [line.split() for line in Path(mountinfo_path).read_text().splitlines()]
    mounted = [r for r in rows if len(r) > 6 and unescape(r[4]) == str(root)]
    if len(mounted) != 1 or 'ro' not in mounted[0][5].split(','):
        raise ValueError('空间内未确认该路径为实际只读挂载，不能用同名缓存目录代替')
    if index_relative:
        rel = PurePosixPath(index_relative)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('索引必须位于本次挂载目录内')
        indexes = [root / rel]
    else:
        indexes = list(root.glob('data_link*.json'))
        if plan.get('evaluation_head') == 'vision':
            # Discover shallow delivered indexes by their declared split, not a
            # hard-coded release filename. Never recurse through frame trees.
            indexes += list(root.glob('*/data_link*.json'))
            indexes += list(root.glob('*/*/data_link*.json'))
            if len(indexes) > 64:
                raise ValueError('视觉索引候选过多，需要交付信息明确索引')
            candidates = {}
            for candidate in indexes:
                candidate.resolve().relative_to(root)
                payload = candidate.read_bytes()
                value = json.loads(payload)
                if any(v.get('frames') for v in value.get('vision_eval', {}).values()):
                    candidates.setdefault(hashlib.sha256(payload).hexdigest(), candidate)
            indexes = list(candidates.values())
    if len(indexes) != 1:
        raise ValueError('本次挂载未找到唯一数据索引；有多个时需从交付信息明确索引，不能沿用旧文件名')
    index = indexes[0]
    index.resolve().relative_to(root)
    raw = index.read_bytes()
    data = json.loads(raw)
    counts = {}
    for key in ('eval', 'joint_eval', 'vision_eval'):
        scenes = data.get(key, {})
        if not isinstance(scenes, dict):
            raise ValueError('数据索引验证集结构无效')
        counts[key] = {'scenes': len(scenes), 'frames': sum(len(v.get('frames', [])) for v in scenes.values())}
    result = {'state': 'MOUNT_VERIFIED', 'asset_binding_sha256': plan['asset']['binding_sha256'],
            'space_uid': plan['space_uid'], 'mount_path': str(root),
            'index_relative': index.relative_to(root).as_posix(),
            'index_sha256': hashlib.sha256(raw).hexdigest(), 'split_counts': counts,
            'execution_location': 'lpai', 'workstation_data_used': False,
            'evaluation_allowed': False, 'next_stage': 'verify_all_head_GT_and_native_adapter'}
    if include_manifest:
        frames = {head:[[scene,f['frame_index']] for scene,info in data.get(split,{}).items() for f in info.get('frames',[])]
                  for head,split in [('occ','eval'),('vision','vision_eval')]}
        for head,rows in frames.items():
            if any(not isinstance(s,str) or not s or not isinstance(f,(str,int)) or isinstance(f,bool) for s,f in rows):
                raise ValueError('验证帧标识无效')
            if len(rows)!=len(set((s,str(f)) for s,f in rows)):
                raise ValueError('验证索引含重复帧')
        signature=hashlib.sha256(json.dumps(frames,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        result['frame_manifest_sha256']=signature
        result['dataset_source']={'id':plan['asset']['version_uid'],'version':plan['asset']['version'],
            'source_ref':plan['asset']['source_ref'],'asset_binding_sha256':plan['asset']['binding_sha256'],
            'index_path':str(index),'index_sha256':result['index_sha256'],
            'frames_by_head':frames,'frame_manifest_sha256':signature,
            'mount_receipt':dict(result)}
        if plan.get('evaluation_head') == 'vision':
            scenes = list(data.get('vision_eval', {}))
            sample = scenes[:8]+scenes[-8:]
            for scene in sample:
                if PurePosixPath(scene).is_absolute() or '..' in PurePosixPath(scene).parts:
                    raise ValueError('视觉索引包含非法场景路径')
            roots = [root, index.parent]
            roots += [p for p in index.parent.parent.iterdir() if p.is_dir()] if index.parent != root else []
            roots = list(dict.fromkeys(p.resolve() for p in roots))
            if len(roots) > 64:
                raise ValueError('视觉数据目录候选过多，需要交付信息明确目录')
            accepted = []
            for candidate in roots:
                candidate.relative_to(root)
                if any((candidate/s/'dataset').is_dir() or (candidate/s/'original').is_dir() for s in sample):
                    accepted.append(str(candidate))
            result['dataset_source']['data_roots'] = accepted or [str(root)]
    return result
