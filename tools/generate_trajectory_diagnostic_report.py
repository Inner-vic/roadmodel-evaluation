"""Atomic, offline diagnostic supplement generation from existing artifacts."""
import json,sys,hashlib,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.model_report import p,h2,table
from pcm_eval.common import load_json,save_json,digest,utc_now

ROOT=Path('evidence/model-evaluation/remaining-heads-20261027/semantic-diagnostic-01')
RUN=ROOT/'trajectory-diagnostic-01'
DEST=Path('evidence/reports/model-trajectory-diagnostic-20261027')
assert not DEST.exists(),'Preserve prior reports'
a=load_json(ROOT/'analysis.json');r=load_json(RUN/'diagnostic-receipt.json');amp=a['full_model_amp_control']
assert r['state']==amp['state']=='succeeded' and a['unique_frames']==2 and a['total_new_full_model_frame_passes']==4
assert all(all(x['original_prediction_exact_match'].values()) for x in a['frames'])
assert all(x['changed_labels_vs_FP32']==0 and x['same_GT'] and x['same_trajectory_xy'] for x in amp['frames'])
for folder,receipt in [(RUN,r),(RUN/'semantic-component-replay-02',a['component_replay']),(ROOT/'trajectory-amp-control-01',amp)]:
    for item in receipt['outputs']:assert digest(folder/item['path'])==item['sha256'],item['path']
assert (ROOT/'run_trajectory_diagnostic_01.py').read_bytes()==Path('tools/run_trajectory_diagnostic_remote.py').read_bytes()
DEST.mkdir()
report={'schema':'roadmodel.trajectory-diagnostic-report/v1','generated_at':utc_now(),
    'source_model_version':'v3.2.2','evaluation_type':'model_accuracy_diagnostic','source_run_id':r['source_inference_run_id'],
    'analysis':a,'formal_metric_update':False,'quality_assessment':'not_assessed',
    'conclusions':{
        'confirmed':['Both frames exactly reproduce all retained native trajectory outputs.',
            'All valid points strongly prefer asphalt in real logits; no NaN, tie, remap or score-fill artifact.',
            'Feature term dominates final asphalt-minus-cement margin; classifier bias alone does not explain it.',
            'Gaussian conditioning reinforces asphalt; removing it does not remove the error. Position-only also predicts asphalt on this scope.',
            'LiDAR features are present and spatially varying; removing LiDAR strengthens asphalt preference.',
            'Training-style full-model FP16 autocast yields no changed semantic labels on the same two frames.'],
        'scope':'Inference mechanism on two selected frames; not a model-wide or training-causality claim.',
        'unverified':['Training-class distribution, supervision balance and training convergence causes were not established.',
            'Feature ablations are off-distribution diagnostic interventions and not deployment fixes or quality scores.']},
    'generation':{'existing_artifacts_only':True,'new_inference_frames':0,'PCM_release_template_used':False}}
save_json(DEST/'report.json',report)
out=[h2('同两帧 logits 与中间特征诊断'),p('本轮已补齐原来缺少的 logits、概率和中间特征。结论已定位到当前权重下的语义条件融合与决策：高斯条件和位置特征经语义网络后形成强沥青倾向，语义上下文进一步放大分差；LiDAR 信号没有缺失，但未能扭转最终分类。未发现评测标签、GT、权重接入或输出解析错误。')]
out.append(table(['项目','执行与复现'],[
    ['唯一输入','同一场景的原 20、21 两帧，权重、配置、GT 与坐标范围保持一致'],
    ['新增整网前向','FP32 两帧次 + 训练同款 FP16 autocast 两帧次，共 4 帧次；唯一帧数仍为 2'],
    ['模块诊断','仅对留存特征重放语义模块及高斯条件分解，未新增整网前向或训练'],
    ['复现检查','FP32 的标签、高度、轨迹 XY、有效掩码和 GT 与原 NPZ 逐项完全相同；全部 logits 与关键特征为有限值'],
    ['精度对照','FP16 整网相对 FP32 标签变化 0/1360，GT 与轨迹 XY 相同；FP16/BF16 语义模块重放也未改变分类']],[180,580]))
out.append('<h3>实际 logits 与概率</h3>')
out.append(table(['帧','有效 / 正确','沥青概率中位数（%）','水泥概率中位数（%）','沥青−水泥平均 logit 分差'],[
    [x['frame'],f"{x['valid_points']} / {x['correct']}",x['asphalt_probability']['0.5']*100,x['cement_probability']['0.5']*100,x['mean_logit_margin']] for x in a['frames']],[70,110,170,170,240]))
out.append(p('这里是模型输出的 softmax 概率，不是准确率。两帧沥青概率最低值分别为 '+
    '、'.join(f"{x['asphalt_probability']['0']*100:.2f}%" for x in a['frames'])+'，均没有接近平票；实际准确率仍为 0/1360。'))
out.append('<h3>分差来自哪里</h3>')
out.append(table(['帧','特征项贡献','bias 项贡献','最终平均分差'],[[x['frame'],x['mean_feature_margin_contribution'],x['bias_margin_contribution'],x['mean_logit_margin']] for x in a['frames']],[80,240,220,220]))
out.append(p('最后分类器是线性层，可精确拆成特征项与 bias 项；独立重建 logits 的最大误差小于 0.000007。约 8.08 / 8.57 的分差主要由特征项形成，不能归因为某个固定 bias 把输出写成沥青。'))
out.append(table(['固定最终分类器对特征的诊断投影','帧 20：沥青−水泥平均分差','帧 21：沥青−水泥平均分差'],[
    [label,a['frames'][0]['semantic_stages'][i]['mean_asphalt_minus_cement_margin'],a['frames'][1]['semantic_stages'][i]['mean_asphalt_minus_cement_margin']]
    for i,label in enumerate(['语义上下文输入','上下文块 1 后','上下文块 2 后','上下文块 3 后','上下文块 4 后'])],[300,230,230]))
out.append(p('上表用同一个最终分类器观察各层特征，并非模型原生的中间层分类输出，也不是每层精度。它说明上下文变换在这两帧持续放大了沥青相对水泥的优势。'))
out.append('<h3>留存特征上的条件分支干预</h3>')
names=[('native_replay','原始条件'),('no_gaussian','去掉高斯条件'),('no_lidar','去掉 LiDAR 条件'),
       ('position_only','仅位置条件'),('gaussian_only','仅高斯条件'),('lidar_only','仅 LiDAR 条件'),('zero_input','全零融合输入')]
out.append(table(['诊断条件','帧 20：沥青点 / 水泥点','帧 21：沥青点 / 水泥点','两帧沥青平均概率（%）'],[
    [label]+[f"{x['semantic_interventions'][key]['prediction_counts'].get('1',0)} / {x['semantic_interventions'][key]['prediction_counts'].get('2',0)}" for x in a['frames']]+[
        ' / '.join(f"{x['semantic_interventions'][key]['mean_asphalt_probability']*100:.2f}" for x in a['frames'])]
    for key,label in names],[220,170,170,200]))
out.append(p('高斯条件明显强化沥青倾向：去掉后平均概率由 99.87% / 99.95% 降至 94.82% / 94.61%，但仍全部误判；仅位置条件也全部判沥青，因此不能把高斯分支当成唯一原因。去掉 LiDAR 后反而更偏向沥青；LiDAR 有效查询命中为 '+
    '、'.join(f"{x['valid_surface_lidar_hits']}/{x['valid_surface_queries']}" for x in a['frames'])+'，且特征非零并随位置变化。全零融合输入则全部变成水泥，说明语义模块会随输入变化，并非输出常量。'))
out.append(p('将高斯条件进一步拆为全局与局部项后，单独移除任一项仍全部判沥青。两项都强化当前语义决策，未找到“关闭某一项即可修复”的证据。详细概率与分差在结构化附件中。'))
out.append(p('分支干预会改变正常特征分布，只用于定位影响；表中点数不是正式精度评估或上线改法。实际评测结果始终使用完整模型输出。'))
out.append('<h3>结论与后续边界</h3>')
out.append(p('本轮可确认：这两帧水泥路面落入了当前模型很强的沥青决策区域；偏向来自条件特征及其经过语义上下文后的表示，非标签错位、零值兜底、bias 单项或本次 FP32/FP16 切换造成。高斯条件增强了该倾向，位置条件也具有明显影响，LiDAR 未能扭转。'))
out.append(p('尚未证明造成该权重行为的训练原因。类别比例、语义监督强度、训练收敛或场景分布等需要原训练/验证统计才能判断，不能凭两帧断言训练塌缩。下一步应优先对照该权重的水泥场景训练/验证表现与条件融合设计；不需要通过全量评测才能继续排查。'))
out.append(p('已保存：两帧全类别 logits/概率、位置/高斯/LiDAR 特征、4 个语义上下文块输出、语义分类权重、实际解析配置、各干预输出、FP16 整网对照、逐点 CSV 及哈希收据。评测、分析生成与飞书发布分别调用，未耦合 PCM 发版报告。'))
inputs=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ['.npz','.json','.log','.py','.csv']]
inputs.extend(Path('tools')/n for n in ['run_trajectory_diagnostic_remote.py','replay_retained_gaussian_components_remote.py','run_trajectory_amp_control_remote.py','analyze_trajectory_diagnostic.py'])
with zipfile.ZipFile(DEST/'evidence.zip','x',zipfile.ZIP_DEFLATED) as archive:
    manifest=[]
    for i,path in enumerate(dict.fromkeys(inputs)):
        assert path.stat().st_size<16*1024*1024
        name=f'{i:02d}-{path.name}';archive.write(path,name);manifest.append({'name':name,'source':str(path),'sha256':digest(path)})
    archive.writestr('manifest.json',json.dumps(manifest,indent=2))
for path,label in [(DEST/'report.json','轨迹诊断补跑结果.json'),(DEST/'evidence.zip','轨迹logits与中间特征证据.zip')]:
    out.append('<source path="@./'+path.as_posix()+'" name="'+label+'"/>')
(DEST/'report.xml').write_text(''.join(out),encoding='utf-8')
print(json.dumps({'state':'generated','output':str(DEST),'unique_frames':2,'source_full_frame_passes':4,'new_inference_during_generation':0,'evidence_bytes':(DEST/'evidence.zip').stat().st_size}))
