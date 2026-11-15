"""Model single-report rendering: existing results in, XML/JSON out; no execution or publication."""
from collections import defaultdict
from copy import deepcopy
from html import escape
import math
from pathlib import Path
import zipfile
from .common import load_json,save_json,digest,utc_now


def build_model_report(result, *, binding=None, gt_preparation=None,head_results=None,trajectory_figure=None,head_statuses=None):
    if result.get('schema')!='pcm.single-eval/v1' or result.get('evaluation_type')!='model_accuracy':
        raise ValueError('One model_accuracy result is required')
    data=deepcopy(result);metrics=data.get('metrics',[])
    if not metrics:raise ValueError('No existing model metrics to report')
    for m in metrics:
        if type(m.get('value')) not in (int,float) or not math.isfinite(m['value']):
            raise ValueError('Metrics must be finite numbers')
    prov=data.get('provenance',{});run=prov.get('native_run',prov.get('lpai_task',{}))
    if binding and run.get('checkpoint_sha256'):
        for field in ['main_checkpoint','classifier_checkpoint']:
            item=binding[field]
            if item['sha256']!=run['checkpoint_sha256'].get(item['filename']):raise ValueError('Checkpoint binding mismatch')
        if binding['dataset']['selected_frames']!=run.get('selected_frames'):raise ValueError('Dataset scope mismatch')
    if gt_preparation and run.get('selected_frames'):
        selected=[[f['scene'],f['frame_index']] for f in gt_preparation.get('frames',[])]
        if gt_preparation.get('state')!='validated' or selected!=run['selected_frames']:
            raise ValueError('GT preparation describes a different frame scope')
    counts=defaultdict(dict)
    for m in metrics:
        d=m.get('dimensions',{})
        if d.get('scope') in ('EVAL','EVAL_PILLAR') and d.get('class_id'):
            k=(d['scope'],d['class_id']);row=counts[k]
            row.update(scope=d['scope'],class_id=d['class_id'],class_name=d['class_name'])
            row[m['name'].removeprefix('occ.')]=m['value']
            row.setdefault('source',m.get('source'))
    sections=[]
    for scope in ('EVAL','EVAL_PILLAR'):
        rows=[v for (s,_),v in counts.items() if s==scope]
        for row in rows:
            s,c,p=[row.get(k) for k in ('seen','correct','positive')]
            if any(type(x)!=int for x in (s,c,p)) or min(s,c,p)<0 or c>min(s,p):
                raise ValueError('Invalid model class counts')
            row.update(fp=p-c,fn=s-c,coverage='with_gt' if s else 'no_gt',
                       exact_iou=c/(s+p-c)*100 if s+p-c else None)
            if s and abs(row['exact_iou']-row['iou'])>.011:raise ValueError('Class IoU differs from counters')
        semantic=[r for r in rows if r['class_id']!='NN'];present=[r for r in semantic if r['seen']>0]
        if semantic:
            sections.append({'scope':scope,'classes':semantic,'non_empty':next((r for r in rows if r['class_id']=='NN'),None),
              'with_gt':len(present),'without_gt':len(semantic)-len(present),
              'present_miou':sum(r['exact_iou'] for r in present)/len(present) if present else None,
              'present_miou_derivation':'mean(correct/(seen+positive-correct)*100 for semantic classes with seen>0)'})
    heads={};all_metrics=deepcopy(metrics)
    for extra in head_results or []:
        if extra.get('schema')!='pcm.single-eval/v1' or extra.get('evaluation_type')!='model_accuracy':raise ValueError('Model head result required')
        ep=extra.get('provenance',{});er=ep.get('native_run',{});kind=er.get('kind')
        if kind not in ('vision','trajectory') or kind in heads:raise ValueError('One result per additional model head required')
        if extra.get('version_name')!=data.get('version_name') or er.get('checkpoint_sha256')!=run.get('checkpoint_sha256') or er.get('source_commit')!=run.get('source_commit'):
            raise ValueError('Additional head must use the same model version, checkpoints and source')
        count=extra.get('coverage',{}).get('observed',{}).get('model_samples')
        if not er.get('run_id') or not count or len(er.get('selected_frames',[]))!=count or len(set(map(tuple,er['selected_frames'])))!=count:
            raise ValueError('Additional head has inconsistent sample scope')
        if extra.get('collection',{}).get('status')!='complete' or extra.get('execution',{}).get('state')!='succeeded':raise ValueError('Complete additional head result required')
        if not extra.get('metrics'):raise ValueError('Additional head has no metrics')
        for m in extra['metrics']:
            if not m['name'].startswith(kind+'.') or type(m['value']) not in (int,float) or not math.isfinite(m['value']):raise ValueError('Invalid additional head metrics')
        heads[kind]=deepcopy(extra)
        all_metrics.extend(dict(deepcopy(m),evaluation_head=kind) for m in extra['metrics'])
    if trajectory_figure:
        tr=heads.get('trajectory',{}).get('provenance',{}).get('native_run',{})
        if trajectory_figure.get('source_run_id')!=tr.get('source_inference_run_id',tr.get('run_id')) or [x['frame'] for x in trajectory_figure.get('frames',[])]!=[fr for _,fr in tr.get('selected_frames',[])]:raise ValueError('Trajectory figure scope mismatch')
    return {'schema':'roadmodel.model-single-report/v1','evaluation_type':'model_accuracy','generated_at':utc_now(),
      'version_name':data.get('version_name'),'run_id':run.get('run_id') or run.get('task_uid'),
      'execution':data.get('execution',{}),'coverage':data.get('coverage',{}),
      'collection':data.get('collection',{}),'quality_assessment':'not_assessed',
      'run':run,'model_preflight':prov.get('model_preflight',{}),'binding':deepcopy(binding or {}),
      'gt_preparation':deepcopy(gt_preparation or {}),'occ_sections':sections,
      'head_statuses':deepcopy(head_statuses or {}),
      'vision_metrics':[m for m in all_metrics if m['name'].startswith('vision.')],
      'trajectory_metrics':[m for m in all_metrics if m['name'].startswith(('trajectory.','scheme.'))],
      'diagnostics':[m for m in metrics if m['name'].startswith('gaussian.')],
      'all_metrics':all_metrics,'source_result':data,'head_results':heads,'trajectory_figure':deepcopy(trajectory_figure or {}),
      'generation':{'mode':'existing_results_only','new_inference_frames':0,'release_template_required':False}}


def p(s):return '<p>'+escape(str(s))+'</p>'
def h1(s):return '<h1>'+escape(s)+'</h1>'
def h2(s):return '<h2>'+escape(s)+'</h2>'
def f(v):return '—' if v is None else f'{v:.2f}' if type(v)==float else str(v)
def table(headers,rows,widths=None):
    cols='<colgroup>'+''.join(f'<col width="{w}"/>' for w in widths)+'</colgroup>' if widths else ''
    return '<table>'+cols+'<thead><tr>'+''.join('<th background-color="light-gray">'+p(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+p(f(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def render_model_xml(report, *, json_attachment=None,bundle_attachment=None):
    r=report;run=r['run'];b=r['binding'];pre=r['model_preflight'];env=run.get('environment',{})
    by_name={m['name']:m for m in r['all_metrics'] if not m.get('dimensions',{}).get('scope')}
    metric=lambda name:by_name.get(name,{}).get('value')
    main=b.get('main_checkpoint',{}).get('filename','主权重')
    classifier=b.get('classifier_checkpoint',{}).get('filename','分类头权重')
    out=['<title>RoadModel 模型精度单次评测报告 '+escape(r.get('version_name') or '')+'</title>']
    out+=[h1('本次评测概览'),table(['项目','本次结果'],[
      ['执行状态',r['execution'].get('state','unknown')],['运行 ID',r['run_id']],
      ['范围',f"{len({s for s,_ in run.get('selected_frames',[])})} 个场景 / {r['coverage'].get('observed',{}).get('model_samples','未知')} 帧"],
      ['执行耗时（含初始化）',f"{run.get('duration_seconds',0):.2f} 秒"],
      ['已验证任务','OCC 体素分类、柱状分类、可行驶表面' if r['occ_sections'] else '见下方任务结果'],
      ['质量结论','未配置业务门槛；本报告仅描述本次样本结果']],[190,570])]
    out.append(p('本次为小场景链路验证。各任务分别展示，不合成总分；小样本分数不代表全量精度，也不用于版本优劣比较。'))
    if r.get('head_results'):
        out[2]=table(['项目','本次结果'],[['执行状态','各分任务结果已收集；未设置业务质量门限'],['模型版本',r['version_name']],
            ['验证范围','OCC、视觉分类、轨迹分别使用独立样本范围，见下表'],['质量判定','未判定，不合成总分']],[190,570])
        out.append(table(['分任务','数据版本','场景 / 帧数','运行 ID'],[
            ['OCC',b.get('dataset',{}).get('version',''),f"{len({s for s,_ in run.get('selected_frames',[])})} / {len(run.get('selected_frames',[]))}",r['run_id']]]+
            [[{'vision':'视觉分类','trajectory':'轨迹'}[kind],x['provenance']['native_run'].get('dataset_version',''),
              f"{len({s for s,_ in x['provenance']['native_run']['selected_frames']})} / {len(x['provenance']['native_run']['selected_frames'])}",x['provenance']['native_run']['run_id']]
             for kind,x in r['head_results'].items()],[110,120,110,420]))
    out+=[h1('评测对象与配置'),table(['项目','配置 / 来源'],[
      ['模型版本',r['version_name']],['权重组合',f'{main} 整网 + {classifier} 的 classifier.* 覆盖' if run.get('checkpoint_sha256') else '见来源收据'],
      ['固定源码',run.get('source_commit',b.get('code',{}).get('evaluation_commit','未提供'))],
      ['配置文件',b.get('code',{}).get('config','见原始配置')],
      ['OCC 数据集',b.get('dataset',{}).get('name','')+' / '+b.get('dataset',{}).get('version','')],
      ['运行环境',f"PyTorch {pre.get('torch','未提供')} / CUDA runtime {pre.get('cuda','未提供')}；GPU 索引 {env.get('CUDA_VISIBLE_DEVICES','见收据')}"],
      ['加载检查',f"整网 {pre.get('checkpoint',{}).get('checkpoint_keys','未提供')} 个键；分类头覆盖 {pre.get('classifier_overlay_keys','未提供')} 个键；形状检查见附件"],
      ['OCC 推理方式',f"HEAD_MODE={env.get('HEAD_MODE','见配置')}；TRAIN_MODE={env.get('TRAIN_MODE','见配置')}；FP32；batch={env.get('BATCH_SIZE','见配置')}"],
      ['模型 ROI / 网格',f"{pre.get('model',{}).get('pc_range','见配置')} / {pre.get('model',{}).get('HWD','见配置')}" ]],[190,570])]
    if b.get('model_reference_url'):out.append('<p>模型来源：<a href="'+escape(b['model_reference_url'])+'">原发版报告所附模型记录</a></p>')
    out+=[h1('样本覆盖与输入适配'),p('以下为 OCC 样本与 GT 适配。视觉和轨迹样本分别列于各自章节。'),table(['场景','帧索引'],[[s,', '.join(str(fr) for ss,fr in run.get('selected_frames',[]) if ss==s)] for s in dict.fromkeys(s for s,_ in run.get('selected_frames',[]))],[580,180])]
    if r['gt_preparation']:
        gt=r['gt_preparation'];out.append(p('GT 由 V161 网格映射到模型 ROI：源原点 [5,-3.6,-2]，目标原点 [5,-3.6,-1]，体素间距保持 [0.05,0.15,0.02] 米。Z 索引平移 -50，裁剪范围外的点；保留点的物理坐标、类别、可行驶标签均未改变。'))
        out.append(table(['帧','原始 GT 点','保留点','裁剪点'],[[x['frame_index'],x['source_rows'],x['kept_rows'],x['cropped_rows']] for x in gt['frames']],[100,220,220,220]))
    if b.get('dataset',{}).get('note'):out.append(p(b['dataset']['note']))
    out.append(p('OCC 指标来自整批汇总；当前归一化结果未提供逐帧分数。'))
    out.append(h1('OCC 与柱状结果'))
    for sec in r['occ_sections']:
        pillar=sec['scope']=='EVAL_PILLAR';name='柱状分类' if pillar else 'OCC 体素分类'
        out+=[h2(name),table(['统计量','结果'],[['有 GT 类别 mIoU（%）',sec['present_miou']],
          ['非空 IoU（%，源码 iou2）',metric('pillar.iou2' if pillar else 'occ.iou2')],
          ['类别覆盖',f"有 GT {sec['with_gt']} 类 / 无 GT {sec['without_gt']} 类"]],[380,380])]
        present=[row for row in sec['classes'] if row['seen']]
        out.append(table(['类别','GT 数','IoU（%）','Precision（%）','Recall（%）'],[[x['class_name'],x['seen'],x['exact_iou'],x['precision'],x['recall']] for x in present],[220,120,140,140,140]))
        out.append(h2(name+'计数明细'))
        out.append(table(['类别','GT / seen','预测 / positive','正确 / correct','FP','FN'],[[x['class_name'],x['seen'],x['positive'],x['correct'],x['fp'],x['fn']] for x in present],[220,108,108,108,108,108]))
        absent=[x['class_name'] for x in sec['classes'] if not x['seen']]
        out.append(p('无 GT 类别（不纳入上方 mIoU）：'+', '.join(absent)+'。原始结果中的零值及完整类别计数保留在 JSON；没有样本不能解释为该类精度为零。'))
    if metric('drivable.iou') is not None:
        out+=[h2('可行驶表面'),table(['指标','IoU（%）','seen','correct','positive'],[['原生 DS-IoU',metric('drivable.iou'),metric('drivable.seen'),metric('drivable.correct'),metric('drivable.positive')]],[240,130,130,130,130]),p('DS-IoU 使用源码中的单层可行驶表面匹配规则，保留原始口径；与非空 IoU 分开展示。')]
    for key,title,items in [('vision','视觉分类结果',r['vision_metrics']),('trajectory','轨迹与表面任务结果',r['trajectory_metrics'])]:
        out.append(h1(title))
        if not items:out.append(p('本次未评测：'+({'vision':'所选两帧的 OCC 清单没有帧级分类 GT。分类头已严格加载，此处不填零分。','trajectory':'首轮明确使用 HEAD_MODE=gs_only，轨迹头不参与本次推理计分。'}[key])))
        else:
            from .model_report_heads import render_vision,render_trajectory
            out.append(render_vision(r) if key=='vision' else render_trajectory(r))
    out+=[h1('指标口径与附加统计'),p('每类 IoU = correct / (seen + positive − correct)。有 GT 类别 mIoU 由 seen>0 的类别计数重新计算。源码原始 mIoU 将无 GT 类别按 100% 纳入，以下保留原值用于追溯；不把它作为小场景概览分数。'),
      table(['源码原始统计','值（%）'],[['OCC mIoU',metric('occ.miou')],['柱状 mIoU',metric('pillar.miou')]], [380,380])]
    if r['diagnostics']:out.append(h2('高斯不透明度统计')+table(['统计项','值','单位'],[[m['name'].removeprefix('gaussian.opacity.'),m['value'],m['unit']] for m in r['diagnostics']],[380,200,180]))
    out+=[h1('明细与复现资料'),p(f"已解析 {len(r['all_metrics'])} 个指标记录。报告 JSON 保留全部指标、来源行号和原始归一化结果；证据包保留原始日志、配置、执行及输入校验收据。报告生成本身不触发评测，也不依赖 PCM 发版模板。")]
    for path,label in [(json_attachment,'模型精度单次报告.json'),(bundle_attachment,'模型精度评测证据.zip')]:
        if path:out.append('<source path="@./'+escape(Path(path).as_posix())+'" name="'+label+'"/>')
    return ''.join(out)


def write_model_report(output, *, result_file,binding_file=None,gt_preparation_file=None,head_result_files=None,
                        trajectory_figure_file=None,supporting_artifacts=None,layout='business',head_statuses=None):
    if layout not in ('business','diagnostic'):raise ValueError('Unknown report layout')
    dest=Path(output).resolve()
    if dest.exists():raise ValueError('Choose a new report directory; previous reports are preserved')
    result=load_json(result_file)
    heads=[load_json(path) for path in head_result_files or []]
    from .validation import require_result
    for item in [result]+heads:require_result(item)
    figure=load_json(trajectory_figure_file) if trajectory_figure_file else None
    if figure:
        declared={a['sha256'] for x in heads for a in x.get('artifacts',[]) if a['role']=='retained_trajectory_prediction'}
        if not figure.get('source_files') or {a['sha256'] for a in figure['source_files']}!=declared:raise ValueError('Figure prediction sources differ from trajectory result')
        for a in figure['source_files']:
            if digest(Path(a['path']))!=a['sha256']:raise ValueError('Figure prediction hash mismatch')
    report=build_model_report(result,binding=load_json(binding_file) if binding_file else None,
                              gt_preparation=load_json(gt_preparation_file) if gt_preparation_file else None,
                              head_results=heads,trajectory_figure=figure,head_statuses=head_statuses)
    report['generation']['layout']=layout
    dest.mkdir(parents=True)
    json_file=dest/'report.json';save_json(json_file,report)
    inputs=[Path(result_file)]+[Path(p) for p in (binding_file,gt_preparation_file) if p]
    inputs.extend(Path(p) for p in head_result_files or [])
    inputs.extend(Path(p) for p in supporting_artifacts or [])
    if figure:inputs.extend(Path(p) for p in [trajectory_figure_file,figure['source_data'],figure['image']])
    for x in [result]+heads:
        for a in x.get('artifacts',[]):
            if not Path(a['path']).is_file() or digest(Path(a['path']))!=a['sha256']:raise ValueError('Report source artifact missing or changed: '+a['path'])
            inputs.append(Path(a['path']))
    # Only existing, declared artifacts are bundled; do not crawl model or dataset directories.
    root=Path(result_file).resolve().parent
    inputs.extend(root/n for n in ['native-launch.json','eval-console.log','source-manifest.json','operator-build-state.json'] if (root/n).is_file())
    for extra in head_result_files or []:
        parent=Path(extra).resolve().parent
        inputs.extend(parent/n for n in ['launch.json','eval-console.log','source_inference_receipt.json','trajectory-console.log','trajectory.log','recovery-console.log'] if (parent/n).is_file())
    for path in inputs:
        if path.suffix.lower() in ('.pth','.pt','.ckpt') or path.stat().st_size>64*1024*1024:raise ValueError('Report evidence excludes checkpoint or bulk data files')
    unique=list(dict.fromkeys(p.resolve() for p in inputs));manifest=[]
    with zipfile.ZipFile(dest/'evidence.zip','x',zipfile.ZIP_DEFLATED) as z:
        for i,path in enumerate(unique):
            name=f'{i:02d}-{path.name}';z.write(path,name);manifest.append({'name':name,'sha256':digest(path),'source':str(path)})
        z.writestr('manifest.json',__import__('json').dumps(manifest,indent=2))
    xml=render_model_xml(report,json_attachment=json_file.name,bundle_attachment='evidence.zip')
    if layout=='business':
        from .business_layout import render_model_business
        xml=render_model_business(report,xml)
    (dest/'report.xml').write_text(xml,encoding='utf-8')
    return {'schema':report['schema'],'report_json':str(json_file),'report_xml':str(dest/'report.xml'),
            'evidence_bundle':str(dest/'evidence.zip'),'run_id':report['run_id'],'metric_records':len(report['all_metrics']),
            'new_inference_frames':0}
