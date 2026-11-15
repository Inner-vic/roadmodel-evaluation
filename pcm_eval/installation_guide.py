"""Guided, read-only deployment configuration; never create or start a space."""
from copy import deepcopy
import json
from pathlib import Path
import re

from .common import load_json, save_json
from .installation import validate_request
from .installation_cli import verified_feishu_identity
from .lpai import request


def template_from_space(raw):
    """Copy resource choices, not the source account, UID, node or credentials."""
    meta, spec = raw['metadata'], raw['spec']
    annotations, labels = meta.get('annotations', {}), meta.get('labels', {})
    resources = deepcopy(spec['resources'])
    # A reference notebook's node is not a portable deployment requirement.
    resources.get('affinity', {}).pop('kubernetes.io/hostname', None)
    hosts = set()
    for access in spec.get('accesses', []):
        if access.get('name') == 'ssh':
            match = re.search(r'\b[A-Za-z0-9_-]+@([A-Za-z0-9.-]+)\b', access.get('address', ''))
            if match:
                hosts.add(match.group(1))
    if len(hosts) != 1:
        raise ValueError('参考空间缺少唯一的官方 SSH 入口，请改用审核过的模板文件')
    return {'id': 'roadmodel-selected-gpu-v1', 'namespace': meta['namespace'],
            'cluster': spec['clusterName'], 'project': annotations.get('lpai.org/gems-project'),
            'resource_queue': resources.get('resourceQueue') or labels.get('lpai.org/resourceQueue'),
            'image': annotations.get('lpai.org/base-image') or spec.get('image'),
            'resources': resources, 'accesses': {'visibility': 'personal', 'ports': [22, 8888]},
            'idle_shutdown': {'enabled': True, 'minutes': 240}, 'ssh_host': hosts.pop()}


def make_request(template, account, ssh, dataset_ref=None):
    template = deepcopy(template)
    template.get('resources', {}).get('affinity', {}).pop('kubernetes.io/hostname', None)
    gpu_type = template.get('resources', {}).get('gpuType', '')
    if not gpu_type or gpu_type.lower() in ('cpu', 'none'):
        raise ValueError('当前模型适配器需要 CUDA GPU，CPU 模式尚未适配；不要求固定 A100 型号')
    value = {'schema': 'roadmodel.installation-request/v1', 'account': account,
             'template': template, 'ssh': ssh}
    if dataset_ref:
        value['dataset_ref'] = dataset_ref
    return validate_request(value)


def add_arguments(parser):
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--from-space', help='参考空间名称；只读取其规格，不复用其账号或空间 UID')
    source.add_argument('--template-file', help='审核过的环境模板 JSON，或含 template 的初始化 JSON')
    parser.add_argument('--namespace');parser.add_argument('--cluster')
    parser.add_argument('--lpai-login');parser.add_argument('--lpai-account')
    parser.add_argument('--ssh-credential');parser.add_argument('--identity-file')
    parser.add_argument('--dataset-ref', help='可选的初始 OCC 数据版本链接；例行评测仍使用卡片输入')
    parser.add_argument('--output', required=True, help='新配置文件路径；不会覆盖现有文件')
    parser.add_argument('--lark-cli', default='lark-cli');parser.add_argument('--lark-profile')
    parser.add_argument('--lpai-cli', default='lpai-agent')
    parser.add_argument('--auth-via-deployment', help='通过已有部署核验飞书身份，不复制令牌')
    parser.add_argument('--non-interactive', action='store_true', help='缺少配置时立即报错，不进入询问')
    return parser


def configure(args, *, ask=input, emit=print, query=None, verify_identity=None):
    if Path(args.output).exists():
        raise ValueError('输出配置已存在，请选择新文件；不会覆盖现有部署')

    def need(value, label):
        if value:
            return value
        if args.non_interactive:
            raise ValueError('缺少配置：'+label)
        answer = ask(label+'：').strip()
        if not answer:
            raise ValueError('缺少配置：'+label)
        return answer

    emit('模型精度使用 CUDA GPU；A100 仅为已验证规格，不是固定要求。其他 GPU 需先验证镜像、算子与显存。')
    if args.template_file:
        source = load_json(args.template_file)
        template = source.get('template', source)
    else:
        name = need(args.from_space, '请选择要参考的 GPU 开发空间名称')
        namespace = need(args.namespace, '该参考空间的 namespace')
        cluster = need(args.cluster, '该参考空间的集群')
        reply = (query or (lambda argv: request(argv, cli=args.lpai_cli)))(
            ['devspaces', 'get', '--name', name, '--namespace', namespace, '--cluster', cluster])
        raw = reply.get('data', reply)
        if (raw.get('metadata', {}).get('name') != name or raw['metadata'].get('namespace') != namespace
                or raw.get('spec', {}).get('clusterName') != cluster):
            raise ValueError('官方返回的参考空间身份与选择不同')
        template = template_from_space(raw)

    if verify_identity:
        identity = verify_identity()
    elif args.auth_via_deployment:
        from tools.evaluation_agent import exchange
        identity = exchange(load_json(args.auth_via_deployment), 'verify_user_identity')
    else:
        identity = verified_feishu_identity(args.lark_cli, args.lark_profile)
        identity = {'app_id': identity['app_id'], 'open_id': identity['open_id']}
    account = {'lpai_login': need(args.lpai_login, '本人的 LPAI SSO 登录名'),
               'lpai_account': need(args.lpai_account, '本人的 LPAI 账号邮箱'),
               'feishu_app_id': identity['app_id'], 'feishu_user_id': identity['open_id']}
    ssh = {'credential_name': need(args.ssh_credential, '本人已有的 LPAI SSH 凭据名称'),
           'identity_file': need(args.identity_file, '执行端已有 SSH 私钥的路径引用（不输入密钥内容）')}
    value = make_request(template, account, ssh, args.dataset_ref)
    summary = {'namespace': template['namespace'], 'cluster': template['cluster'],
               'gpu_type': template['resources']['gpuType'], 'resource_queue': template['resource_queue'],
               'resources': value['template']['resources'], 'image': template['image'],
               'idle_shutdown_minutes': 240}
    emit(json.dumps(summary, ensure_ascii=False, indent=2))
    emit('数据可挂载性、镜像权限和实际 CUDA 兼容性仍须在初始化及小场景验收时核验；本步骤不创建或启动空间。')
    if not args.non_interactive and ask('确认保存以上部署配置？输入 yes：').strip().lower() != 'yes':
        raise ValueError('已取消，未保存配置、创建空间或执行评测')
    save_json(args.output, value)
    emit(json.dumps({'state': 'CONFIGURED', 'request_file': str(Path(args.output).resolve()),
                     'space_created': False, 'dataset_copied': False, 'new_inference_frames': 0}, ensure_ascii=False))
    return 0
