"""Read a delivery record as data, classify its artifacts, and retain provenance.

Nothing read from a record is an executable command or an environment setting.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import parse_qs, urlsplit, urlunsplit

from .common import json_digest


def delivery_url(value):
    if not isinstance(value, str) or len(value) > 1000 or any(c in value for c in '\r\n\x00'):
        raise ValueError('请提供完整的飞书交付链接')
    u = urlsplit(value.strip())
    if (u.scheme != 'https' or u.hostname != 'li.feishu.cn' or u.username or u.password
            or u.port not in (None, 443)):
        raise ValueError('请使用公司飞书交付链接')
    if not re.fullmatch(r'/(?:record/[A-Za-z0-9]+|drive/folder/[A-Za-z0-9]+|base/[A-Za-z0-9]+)/?', u.path):
        raise ValueError('交付链接应指向一条记录、文件夹或明确的Base记录')
    if '/base/' in u.path and not parse_qs(u.query).get('record'):
        raise ValueError('Base交付链接需要定位到具体记录，不能自行选择某个版本')
    return urlunsplit(('https', u.hostname, u.path.rstrip('/'), u.query, ''))


def lpai_links(value, resource):
    """Handle Markdown links and the real record's two concatenated URLs."""
    found = []
    for part in re.findall(r'https://.*?(?=https://|[\s<>\[\]()\"\']|$)', str(value or '')):
        u = urlsplit(part.rstrip('，,;；'))
        match = re.fullmatch(r'/lpaiweb/xspacex_([A-Za-z0-9_-]+)/subApp/'+resource+r'/detail/([A-Za-z0-9_-]+)/?', u.path)
        q = parse_qs(u.query)
        versions = q.get('name' if resource == 'model' else 'version', [])
        if (u.hostname != 'lpai.lixiang.com' or u.username or u.password or u.port not in (None, 443)
                or not match or len(versions) != 1 or not re.fullmatch(r'[A-Za-z0-9_.-]+', versions[0])):
            continue
        canonical = urlunsplit(('https', u.hostname, u.path, u.query, ''))
        row = {'url': canonical, 'namespace': match[1], 'uid': match[2], 'version': versions[0]}
        if row not in found:
            found.append(row)
    return found


def content_values(value, depth=0):
    """Bounded traversal of rich cells and embedded JSON; never execute content."""
    if depth>8:return
    if isinstance(value,dict):
        for item in value.values():yield from content_values(item,depth+1)
    elif isinstance(value,list):
        for item in value:yield from content_values(item,depth+1)
    elif isinstance(value,str):
        yield value
        if len(value)<=1024*1024 and value.lstrip().startswith(('{','[')):
            try:parsed=json.loads(value)
            except ValueError:return
            yield from content_values(parsed,depth+1)


def attachment_rows(value,depth=0):
    if depth>8:return
    if isinstance(value,dict):
        if value.get('name') and any(k in value for k in ('size','file_token','token')):
            yield value;return
        for item in value.values():yield from attachment_rows(item,depth+1)
    elif isinstance(value,list):
        for item in value:yield from attachment_rows(item,depth+1)
    elif isinstance(value,str) and len(value)<=1024*1024 and value.lstrip().startswith(('{','[')):
        try:parsed=json.loads(value)
        except ValueError:return
        yield from attachment_rows(parsed,depth+1)


def _file(row, field, parent=None):
    name = row.get('name', '')
    if (not isinstance(name, str) or not name or '/' in name or '\\' in name
            or name in ('.', '..') or any(c in name for c in '\r\n\x00')):
        raise ValueError('交付附件包含无效文件名')
    size = row.get('size')
    if size is not None and (type(size) is not int or size < 0):
        raise ValueError('交付附件大小无效')
    result = {'name': name, 'size': size, 'field': field}
    for key in ('file_token', 'token', 'sha256'):
        if row.get(key):
            result[key] = row[key]
    if parent:
        result['parent'] = parent
    lower = name.lower()
    if lower.endswith('.onnx'):
        role = 'onnx'
    elif lower.endswith(('.pth', '.pt', '.ckpt')):
        role = 'checkpoint'
    elif lower.endswith(('.tar.gz', '.tgz', '.zip')) and ('packup' in lower or field == '编译产物'):
        role = 'compiled_package'
    elif lower.endswith(('.py', '.yaml', '.yml', '.json', '.md', '.txt', '.sh')):
        role = 'supporting_material'
    else:
        role = 'unclassified'
    result['role'] = 'golden' if field == 'golden' else role
    version = re.search(r'(?<![A-Za-z0-9])v\d+\.\d+\.\d+(?!\d)', name)
    result['version'] = version[0] if version else None
    subnet = re.search(r'_sub(\d+)_', name)
    result['subnet'] = int(subnet[1]) if subnet else None
    result['preferred_packup'] = lower.endswith('_packup.tar.gz')
    return result


def inspect_record(response, source_url):
    if 'ok' in response:
        if response['ok'] is not True:
            raise ValueError('飞书交付记录读取失败')
        response = response['data']
    hint = response.get('hint', {})
    schema = hint.get('fields', {}).get('fields', [])
    cells = hint.get('record_data')
    if cells is not None:
        fields = {r['name']: cells.get(r['id']) for r in schema if r.get('type') != 'not_support'}
    else:
        fields = response.get('fields', {})
    if not isinstance(fields, dict):
        raise ValueError('交付记录没有可读取的字段')
    attachments = []
    for field, values in fields.items():
        for row in attachment_rows(values):
            item=_file(row,field)
            if item not in attachments:attachments.append(item)
    return classify(source_url, fields, attachments, coordinates={
        key: response[key] for key in ('base_token', 'table_id', 'record_id') if response.get(key)})


def classify(source_url, fields, artifacts, *, coordinates=None):
    version = str(fields.get('版本号') or fields.get('版本') or '').strip()
    if version and not re.fullmatch(r'v?\d+(?:\.\d+){1,3}(?:[-+][A-Za-z0-9_.-]+)?', version):
        raise ValueError('交付版本号无效')
    if version and not version.startswith('v'):
        version = 'v'+version
    file_versions = sorted({r['version'] for r in artifacts if r.get('version')})
    issues = []
    if not version:
        if len(file_versions) == 1:
            version = file_versions[0]
        else:
            issues.append('交付材料未确定唯一版本')
    mismatch = [r['name'] for r in artifacts if r.get('version') and r['version'] != version]
    model_links=[];datasets=[];evidence=[]
    for field,value in fields.items():
        for text in content_values(value):
            for resource,target in [('model',model_links),('dataset',datasets)]:
                for row in lpai_links(text,resource):
                    if row not in target:target.append(row)
                    item={'field':field,'resource':resource,'url':row['url'],'rule':'versioned_official_resource_url'}
                    if item not in evidence:evidence.append(item)
    # A conflicting explicit resource is never discarded as prose/history.
    # Existing version/architecture/hash binding must resolve all candidates.
    roles = {role: [r for r in artifacts if r['role'] == role] for role in
             ('onnx', 'compiled_package', 'checkpoint', 'golden', 'supporting_material', 'unclassified')}
    selected, alternatives, compile_issues = [], [], []
    packages = roles['compiled_package']
    subnets = {r['subnet'] for r in packages if r['subnet'] is not None}
    if subnets:
        if subnets != set(range(5)):
            compile_issues.append('五分网交付缺少完整的sub0至sub4编译包')
        for stage in sorted(subnets):
            choices = [r for r in packages if r['subnet'] == stage]
            preferred = [r for r in choices if r['preferred_packup']]
            candidates = preferred or choices
            if len(candidates) != 1:
                compile_issues.append('sub'+str(stage)+'有多个候选编译包，需要核验接口与硬件配置')
            else:
                selected.append(candidates[0])
                alternatives.extend(r for r in choices if r != candidates[0])
    elif len(packages) == 1:
        selected = packages
    elif len(packages) > 1:
        compile_issues.append('存在多个整网编译包，尚未确定本次硬件配置')
    if len(roles['onnx']) != 1:
        compile_issues.append('需要唯一的本次ONNX参考模型'+('，发现多个候选：'+'、'.join(r['name'] for r in roles['onnx']) if roles['onnx'] else ''))
    if not packages:
        compile_issues.append('缺少板端编译产物')
    if mismatch:
        compile_issues.append('附件版本与交付版本不一致：'+'、'.join(mismatch))
    model_issues = [] if model_links or roles['checkpoint'] else ['缺少模型集链接或PTH权重']
    result = {'schema': 'roadmodel.delivery-materials/v1', 'source_url': delivery_url(source_url),
              'version': version, 'coordinates': coordinates or {}, 'artifacts': artifacts,
              'model_assets': model_links, 'dataset_assets': datasets, 'issues': issues,
              'recognition':{'link_evidence':evidence,'file_rule':'file_type_then_verified_artifact_content',
                             'unknown_files':[r['name'] for r in roles['unclassified']]},
              'chains': {
                  'model_accuracy': {'materials_ready': not issues and not model_issues,
                                     'issues': model_issues, 'model_assets': model_links, 'checkpoints': roles['checkpoint']},
                  'compile_alignment': {'materials_ready': not issues and not compile_issues,
                                        'issues': compile_issues, 'onnx': roles['onnx'],
                                        'packages': selected, 'alternatives': alternatives}},
              'golden': roles['golden'], 'training_description': fields.get('描述'),
              'execution_started': False}
    result['materials_sha256'] = json_digest(result)
    return result


class LarkSource:
    def __init__(self, *, executable=None, identity='user', call=None, profile=None):
        # Prefer the native Windows binary; .ps1 is not directly executable by Popen.
        self.executable = executable or shutil.which('lark-cli.exe') or shutil.which('lark-cli') or 'lark-cli'
        candidate = Path(self.executable).parent/'node_modules/@larksuite/cli/bin/lark-cli.exe'
        if not executable and candidate.is_file():
            self.executable = str(candidate)
        if identity not in ('user', 'bot'):
            raise ValueError('Invalid Lark identity')
        self.identity, self.call = identity, call
        if profile is not None and not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',profile):
            raise ValueError('Invalid configured Lark profile')
        self.profile=profile

    def query(self, argv):
        if self.call:
            return self.call(argv)
        options = {'capture_output': True, 'encoding': 'utf-8', 'timeout': 120, 'shell': False,
                   **({'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {})}
        command=[self.executable]+(['--profile',self.profile] if self.profile else [])
        p = subprocess.run([*command, *argv, '--as', self.identity], **options)
        # The installed CLI can still have a refreshable user grant when a
        # business command reports need_user_authorization. Verify/refresh that
        # existing grant once; never start a new login or broaden scopes here.
        if self.identity == 'user' and p.returncode and 'need_user_authorization' in (p.stdout or '')+(p.stderr or ''):
            verified = subprocess.run([*command, 'auth', 'status', '--json', '--verify'], **options)
            if verified.returncode == 0:
                p = subprocess.run([*command, *argv, '--as', self.identity], **options)
        try:
            result = json.loads(p.stdout or p.stderr)
        except ValueError as e:
            raise ValueError('飞书CLI未返回有效JSON，检查网络和已登录身份') from e
        if p.returncode or result.get('ok') is not True:
            error = result.get('error', {})
            raise ValueError('飞书材料读取失败：'+str(error.get('message', error.get('type', 'unknown')))[:300])
        return result

    def read(self, url):
        url = delivery_url(url)
        if '/drive/folder/' not in url:
            return inspect_record(self.query(['base', '+url-resolve', '--url', url]), url)
        token = urlsplit(url).path.rsplit('/', 1)[-1]
        # Read metadata only; do not download all folder contents or run supplied scripts.
        queue, seen, files = [(token, 0)], set(), []
        while queue:
            parent, depth = queue.pop(0)
            if parent in seen:
                continue
            if depth > 8 or len(seen) >= 100:
                raise ValueError('交付文件夹超过目录解析范围，需要明确交付子目录')
            seen.add(parent)
            page = None
            while True:
                params = {'folder_token': parent, 'page_size': 200}
                if page:
                    params['page_token'] = page
                response = self.query(['drive', 'files', 'list', '--params', json.dumps(params)])['data']
                for row in response.get('files', []):
                    if row.get('type') == 'folder':
                        queue.append((row['token'], depth+1))
                    elif row.get('type') == 'file':
                        files.append(_file(row, 'folder', parent))
                    else:
                        files.append({'name': row.get('name'), 'role': 'unclassified', 'token': row.get('token'),
                                      'version': None, 'subnet': None, 'preferred_packup': False, 'field': 'folder'})
                if len(files) > 5000:
                    raise ValueError('交付文件数超过解析范围')
                if not response.get('has_more'):
                    break
                next_page = response.get('next_page_token') or response.get('page_token')
                if not next_page or next_page == page:
                    raise ValueError('文件夹分页未完整返回')
                page = next_page
        return classify(url, {}, files)
