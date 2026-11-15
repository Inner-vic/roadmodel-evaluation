"""Publish an existing offline audit into the same model report, without evaluation."""
import json,sys,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.model_report import p,h2,table

ROOT=Path('evidence/model-evaluation/remaining-heads-20261027/semantic-audit')
PUB=ROOT/'publication';PUB.mkdir(exist_ok=True)
DOC='DEMO00000000000000000000008'
CLI=['D:/nodejs/node.exe','C:/Users/developer/AppData/Roaming/npm/node_modules/@larksuite/cli/scripts/run.js']
def call(name,args):
    run=subprocess.run(CLI+args+['--as','user','--format','json'],capture_output=True,timeout=120)
    (PUB/(name+'.json')).write_bytes(run.stdout)
    if run.stderr:(PUB/(name+'.stderr.txt')).write_bytes(run.stderr)
    assert not run.returncode,name+' CLI failed'
    value=json.loads(run.stdout.decode('utf-8-sig'));assert value.get('ok'),value.get('error')
    return value['data']['document'] if args[:2]==['docs','+fetch'] else value['data']
def tree(d):
    root=ET.fromstring('<doc>'+d['content']+'</doc>')
    return root[0] if len(root)==1 and root[0].tag=='fragment' else root

audit=json.loads((ROOT/'audit.json').read_text(encoding='utf-8'))
assert all(audit['checks'].values()) and not audit['configuration']['model_config_differences_from_training']
payload=h2('轨迹语义 0% 核查')+p('核查结论：在现有两帧证据可核查的范围内，0% 是模型将水泥预测为沥青的实际结果；没有发现标签映射、GT 采样、计分或权重配置接入错误。该结论不等于已定位模型内部误分类的原因，也不能推断全量表现。')
payload+=table(['帧索引','有效点','GT','预测','正确点'],[[x['frame'],x['valid_points'],'2 / 水泥','1 / 沥青',x['correct']] for x in audit['frames']],[100,110,190,190,110])
payload+=table(['核查项','复核结果'],[
    ['标签体系','轨迹 / OCC：0=ignore、1=沥青、2=水泥、26=empty。视觉 basic_road_surface 则为 0=沥青、1=水泥；本次未混用视觉编号。'],
    ['计分','直接比较 traj_label 与 dp3_gt_traj_label，正确 0 / 有效 1360 = 0%。后轮及无效点未参与计分。'],
    ['原始 GT','独立从原始 V151 Mesh 按留存轨迹 XY 重新采样，1360/1360 标签与留存 GT 相同；对应 OCC 柱也全部包含水泥标签，GT 高度坐标核验通过。原始输入哈希与下载收据一致。'],
    ['权重','使用原模型链接 epoch_44.pth，整网 1263 键严格加载，含 164 个轨迹头键；轨迹语义层权重非零且有限。epoch_98.pth 仅覆盖 classifier.* 的 216 个键，不覆盖轨迹头。'],
    ['实际执行分支','TRAIN_MODE=occ、HEAD_MODE=traj_only。本次不执行视觉分类分支；轨迹标签由轨迹头语义 logits 的 argmax 产生。'],
    ['有效配置','与原轨迹任务解析后的 model 配置逐字段一致。Y_OFFSETS 缺省解析为 [-0.105,0,0.105]，与训练一致；LiDAR intensity 等缺省值也一致。NUM_FRAMES/stride 改为 1 用于选取小样本，未启用时序状态，数据加载每次仍取当前帧。'],
    ['原始结果完整性','留存 NPZ 与首次推理保存后的哈希完全相同。NaN 问题发生在无事件分位数的 JSON 序列化阶段，恢复统计未改预测标签、未补造零值、未新增推理。']],[180,580])
payload+=p('证据边界：本次原始文件没有保存语义 logits / 概率及中间特征，因此尚不能区分语义层、上游特征或数值精度等更深层原因。训练使用 AMP、本次评测使用 FP32，这一差异未做控制实验。后续若需定位，应仅在相同两帧补存 logits 和关键特征，不需要跑全量。')
payload+=p('本次核查只读原始结果和输入，新增推理 0 帧。下面附件保留逐点核验、配置差异和哈希记录。')
payload+='<source path="@./'+(ROOT/'audit.json').as_posix()+'" name="轨迹语义0%核查证据.json"/>'
(ROOT/'audit.xml').write_text(payload,encoding='utf-8')

verify='--verify-existing' in sys.argv
outline=json.loads((PUB/'outline.json').read_text(encoding='utf-8-sig'))['data']['document'] if verify else call('outline',['docs','+fetch','--doc',DOC,'--scope','outline','--max-depth','2','--detail','with-ids'])
t=tree(outline);headings=[e for e in t.iter('h1') if ''.join(e.itertext())=='轨迹与表面任务结果'];assert len(headings)==1
section_id=headings[0].get('id')
args=['docs','+fetch','--doc',DOC,'--scope','section','--start-block-id',section_id,'--detail','full']
before=json.loads((PUB/'before.json').read_text(encoding='utf-8-sig'))['data']['document'] if verify else call('before',args)
b=tree(before);elements=list(b)
assert '轨迹语义 0% 核查' not in ''.join(b.itertext()),'Audit already exists; preserve it'
front=[i for i,e in enumerate(elements) if e.tag=='h2' and ''.join(e.itertext())=='前沿事件与边界覆盖'];assert len(front)==1
anchor=elements[front[0]-1];assert anchor.tag=='p'
old_images={x.get('src') for x in b.iter('img')};assert len(old_images)==1
change=json.loads((PUB/'update.json').read_text(encoding='utf-8-sig'))['data'] if verify else call('update',['docs','+update','--doc',DOC,'--command','block_insert_after','--block-id',anchor.get('id'),
    '--content','@./'+(ROOT/'audit.xml').as_posix(),'--doc-format','xml','--revision-id',str(before['revision_id'])])
assert change['result']=='success' and not change.get('warnings'),change
after=json.loads((PUB/'after.json').read_text(encoding='utf-8-sig'))['data']['document'] if verify else call('after',args)
a=tree(after)
for e in ET.fromstring('<doc>'+payload+'</doc>'):
    s=''.join(e.itertext())
    if s:assert s in ''.join(a.itertext()),s[:100]
assert old_images=={x.get('src') for x in a.iter('img')}
new_heading=[e for e in a.iter('h2') if ''.join(e.itertext())=='轨迹语义 0% 核查'];assert len(new_heading)==1
receipt={'document_id':DOC,'url':'https://li.feishu.cn/wiki/DEMO00000000000000000000006#'+new_heading[0].get('id'),
    'revision_id':after['revision_id'],'audit_file':str(ROOT/'audit.json'),'new_inference_frames':0,
    'result':'verified','new_attachment_count':len(list(a.iter('source')))-len(list(b.iter('source')))}
(PUB/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,ensure_ascii=True))
