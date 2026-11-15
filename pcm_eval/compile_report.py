"""Independent, read-only single compile-evaluation report generation.

Inputs are evaluation results and their metadata artifacts. No release template,
baseline, inference runner, network connection or document publishing is used.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from html import escape
import math
from pathlib import Path
import zipfile

from .common import digest, json_digest, load_json, save_json, utc_now

SEMANTICS = Path(__file__).resolve().parents[1] / 'configs/compile_output_semantics.json'
TITLES = [('overview','本次评测概览'), ('configuration','评测对象与配置'),
          ('coverage','样本覆盖与结果有效性'), ('tensor','张量一致性'),
          ('gt','相对GT的表现'), ('findings','关注样本与附加验证'),
          ('definitions','指标口径与统计方法'), ('artifacts','明细与复现资料')]


def number(value, name):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('Expected finite numeric '+name)
    return value


def source_record(name, payload, files):
    return {'name':name, 'sha256':files.get(name, {}).get('sha256') or json_digest(payload),
            'path':files.get(name, {}).get('path'), 'hash_kind':'file' if name in files else 'canonical_json'}


def build_compile_report(result, cosine, *, backend_metrics=None, request=None,
                         source_files=None, supplementary=None, semantics=None):
    if result.get('schema')!='pcm.single-eval/v1' or result.get('evaluation_type')!='compile_alignment':
        raise ValueError('One compile_alignment result is required')
    files=source_files or {}; backend_metrics=backend_metrics or {}; request=request or {}
    if request and request.get('kind')!='compile_alignment':
        raise ValueError('Request type differs from report type')
    live=result.get('provenance',{}).get('live_run',{})
    run_id=live.get('run_id') or result.get('execution',{}).get('run_id')
    if not run_id:raise ValueError('Executed run identity is required')
    if request.get('version_name') and request['version_name']!=result.get('version_name'):
        raise ValueError('Request version differs from executed run')
    for key in ('onnx_sha256','packup_sha256'):
        measured=live.get(key)
        if key=='onnx_sha256' and request.get('execution',{}).get('adapter')=='subnet_chain_workstation':
            measured=live.get('source_onnx_sha256')
            expected={x['stage']:x['sha256'] for x in request.get('model',{}).get('subnet_archives',[])}
            observed={x['stage']:x['sha256'] for x in live.get('subnet_archives',[])}
            if set(expected)!={0,1,2,3,4} or expected!=observed or not live.get('onnx_sha256'):
                raise ValueError('Subnet execution artifacts differ from frozen request')
        if request.get('model',{}).get(key) and request['model'][key]!=measured:
            raise ValueError('Request artifact differs from executed run: '+key)
    count=cosine.get('num_frames'); frames=cosine.get('frames',[])
    if type(count)!=int or count<=0 or len(frames)!=count:
        raise ValueError('Comparison frame inventory mismatch')
    selected=live.get('selected_frames',[])
    if selected and len(selected)!=count:raise ValueError('Executed frame scope mismatch')
    expected_ids=[str(s)+'_'+str(f) for s,f in selected]
    actual_ids=[f['frame_id'] for f in frames]
    if len(set(actual_ids))!=count or (expected_ids and actual_ids!=expected_ids):
        raise ValueError('Per-frame comparison identity/order mismatch')
    names=list(cosine.get('tensors',{})); records=[]; issues=[]
    if not names:raise ValueError('No compared outputs')
    frame_rows=[]
    for i,frame in enumerate(frames):
        seen=[]; valid=[]
        for j,row in enumerate(frame.get('outputs',[])):
            name=row['onnx_output_name'];seen.append(name)
            record=dict(row,frame_id=frame['frame_id'],frame_ordinal=i+1,
                        source={'artifact':'cosine_summary.json','locator':f'/frames/{i}/outputs/{j}'})
            if row.get('status')=='ok':
                for key in ('cosine','mean_abs_diff','max_abs_diff'):number(row.get(key),key)
                if not -1.00001<=row['cosine']<=1.00001 or row['mean_abs_diff']<0 or row['max_abs_diff']<row['mean_abs_diff']-1e-6:
                    raise ValueError('Invalid tensor statistic')
                valid.append(row['cosine'])
            else:issues.append({'code':'unavailable_tensor','frame_id':frame['frame_id'],'output':name,'status':row.get('status')})
            records.append(record)
        if len(seen)!=len(set(seen)) or set(seen)!=set(names):raise ValueError('Missing, duplicate or undeclared output')
        if valid:
            mean=sum(valid)/len(valid)
            if not math.isclose(mean,number(frame.get('mean_cosine'),'frame mean'),abs_tol=2e-7):
                raise ValueError('Frame cosine summary differs from detail')
        frame_rows.append({'ordinal':i+1,'frame_id':frame['frame_id'],'compared_outputs':len(valid),
                           'mean_cosine':frame.get('mean_cosine'),'min_cosine':frame.get('min_cosine')})
    metadata=(semantics or load_json(SEMANTICS))['outputs']; outputs=[]
    for name in names:
        values=[r for r in records if r['onnx_output_name']==name and r.get('status')=='ok']
        original=cosine['tensors'][name]
        if len(values)!=original.get('num_frames'):raise ValueError('Per-output frame count differs')
        if not values:continue
        mean=sum(v['cosine'] for v in values)/len(values);worst=min(values,key=lambda v:v['cosine'])
        if not math.isclose(mean,number(original.get('mean_cosine'),'output mean'),abs_tol=2e-7) or not math.isclose(worst['cosine'],number(original.get('min_cosine'),'output min'),abs_tol=2e-7):
            raise ValueError('Per-output cosine summary differs from detail')
        outputs.append({'name':name,**metadata.get(name,{'label':name,'group':'其他输出'}),
                        'frames':len(values),'mean_cosine':original['mean_cosine'],'min_cosine':original['min_cosine'],
                        'mean_frame_mae':sum(v['mean_abs_diff'] for v in values)/len(values),
                        'max_abs_diff':max(v['max_abs_diff'] for v in values),
                        'worst_frame':worst['frame_id'],'worst_frame_ordinal':worst['frame_ordinal'],
                        'compare_dtypes':sorted({v['compare_dtype'] for v in values}),
                        'source':{'artifact':'cosine_summary.json','locator':'/tensors/'+name.replace('~','~0').replace('/','~1')},
                        'derivation':{'mean_frame_mae':'arithmetic mean of available per-frame mean_abs_diff',
                                      'max_abs_diff':'maximum per-frame max_abs_diff'}})
    actual_mean=sum(f['mean_cosine'] for f in frame_rows)/count
    if not math.isclose(actual_mean,number(cosine.get('mean_cosine_all'),'global mean'),abs_tol=2e-7):
        raise ValueError('Global cosine mean differs from frames')
    if not math.isclose(min(f['mean_cosine'] for f in frame_rows),number(cosine.get('min_cosine_all'),'minimum frame mean'),abs_tol=2e-7):
        raise ValueError('Minimum frame mean differs from frames')
    prov=result.get('provenance',{});adapt=prov.get('gt_adaptation')
    gt_summary=[];class_rows=[];count_rows=[]
    if backend_metrics:
        if set(backend_metrics)!={'board','onnx'} or not adapt or adapt.get('state')!='validated':
            raise ValueError('Both GT backends and a validated scope receipt are required')
        if adapt.get('source_run_id')!=run_id or adapt.get('selected_frames')!=selected or adapt.get('num_frames')!=count:
            raise ValueError('GT result scope differs from tensor comparison')
        gt_ids=[str(f['scene'])+'_'+str(f['frame_index']) for f in adapt.get('frames',[])]
        if gt_ids!=actual_ids:raise ValueError('GT frame inventory differs from tensor comparison')
        support={}
        for backend,data in backend_metrics.items():
            if data.get('num_frames')!=count:raise ValueError('GT backend frame count differs')
            rows=data.get('class_support',[])
            if len(rows)!=len({r['class_id'] for r in rows}):raise ValueError('Duplicate GT class')
            support[backend]={r['class_id']:r for r in rows}
            for row_index,row in enumerate(rows):
                s,c,p=[number(row[k],k) for k in ('seen','correct','positive')]
                if any(type(row[k])!=int for k in ('seen','correct','positive')) or min(s,c,p)<0 or c>min(s,p):
                    raise ValueError('Invalid GT class counts')
                if s>0 and not math.isclose(number(row.get('iou'),'class IoU'),c/(s+p-c)*100,abs_tol=1e-8):
                    raise ValueError('GT IoU differs from counts')
                count_rows.append({'backend':backend,**deepcopy(row),'fp':p-c,'fn':s-c,
                                   'source':{'artifact':f'metrics_{backend}.json','locator':f'/class_support/{row_index}'},
                                   'derivation':{'fp':'positive-correct','fn':'seen-correct'}})
            present=[r['iou'] for r in rows if r['seen']>0]
            if not present or not math.isclose(sum(present)/len(present),number(data.get('present_miou'),'present mIoU'),abs_tol=1e-8):
                raise ValueError('present mIoU differs from class counts')
            for metric in ('present_miou','occ_iou','drivable_iou'):
                source=[m for m in result['metrics'] if m['name']==metric and m.get('dimensions',{}).get('backend')==backend]
                if len(source)!=1 or not math.isclose(source[0]['value'],number(data.get(metric),metric),abs_tol=1e-10):
                    raise ValueError('Normalized GT result differs from source')
        if set(support['board'])!=set(support['onnx']):raise ValueError('GT class spaces differ')
        for cid,board in support['board'].items():
            onnx=support['onnx'][cid]
            if board['name']!=onnx['name'] or board['seen']!=onnx['seen']:raise ValueError('GT class identity/count differs across backends')
            class_rows.append({'class_id':cid,'name':board['name'],'seen':board['seen'],
                               'board_iou':board.get('iou'),'onnx_iou':onnx.get('iou'),
                               'board_positive':board['positive'],'onnx_positive':onnx['positive'],
                               'status':'available' if board['seen'] else 'no_gt_samples',
                               'has_predictions':bool(board['positive'] or onnx['positive'])})
        for metric,label in [('present_miou','present mIoU'),('occ_iou','非空 IoU'),('drivable_iou','可行驶区域 IoU')]:
            gt_summary.append({'metric_id':metric,'label':label,'unit':'percent',
                               'onnx':backend_metrics['onnx'][metric],'board':backend_metrics['board'][metric],
                               'frames':count,'status':'available'})
    valid_records=[r for r in records if r.get('status')=='ok']
    fresh=prov.get('output_integrity',{}).get('state')=='fresh_outputs_checked'
    finite=prov.get('input_validation',{}).get('finite_outputs',{})
    invalid_metrics=[m for m in result['metrics'] if m.get('validity','').startswith('invalid')]
    valid=fresh and finite.get('state')=='passed' and not invalid_metrics and not issues and result.get('collection',{}).get('status')=='complete'
    supplements=[]
    for name,data in (supplementary or {}).items():
        if data.get('run_id') and data['run_id']!=run_id:raise ValueError('Diagnostic run identity differs')
        ids={r.get('frame_id') for r in data.get('checks',data.get('frames',[]))}
        if not ids.issubset(set(actual_ids)):raise ValueError('Diagnostic sample not in this evaluation')
        supplements.append({'name':name,'data':deepcopy(data),'scope_frames':len(ids),
                            'source':source_record(name,data,files)})
    scene_counts=Counter(s for s,_ in selected)
    sources=[source_record('normalized_result.json',result,files),source_record('cosine_summary.json',cosine,files)]
    sources.extend(source_record(f'metrics_{b}.json',d,files) for b,d in backend_metrics.items())
    field_coverage=[{'source':'normalized_result.json','fields':'metrics[*]','handling':'attached_and_rendered','count':len(result['metrics']),
                     'notes':'All normalized values retained; class counters displayed in count tables; scalar/output metrics mapped to result sections.'},
                    {'source':'cosine_summary.json','fields':'frames[*].outputs[*]','handling':'normalized_and_attached','count':len(records),
                     'notes':'All records preserved; full output aggregates and selected lowest-cosine examples shown.'},
                    {'source':'metrics_board.json / metrics_onnx.json','fields':'class_support[*]','handling':'displayed','count':len(count_rows)},
                    {'source':'all input JSONs','fields':'fields not selected for a main table','handling':'retained_in_source_bundle',
                     'notes':'Original files retained unchanged; source-bundle hashes are in the manifest. No claim that every raw field is a semantic metric.'}]
    return {'schema':'roadmodel.compile-single-report/v3','evaluation_type':'compile_alignment',
            'generated_at':utc_now(),'identity':{'run_id':run_id,'version_name':result.get('version_name'),
            'model_version':(result.get('version_name') or '').removeprefix('supplied-reference-'),
            'selected_frames_sha256':json_digest(selected),'onnx_sha256':live.get('onnx_sha256'),
            'packup_sha256':live.get('packup_sha256'),'data_link_sha256':live.get('data_link_sha256'),
            'inference_started_at':live.get('started_at'),'inference_finished_at':live.get('finished_at'),
            'gt_calculated_at':(adapt or {}).get('created_at')},
            'execution':deepcopy(result.get('execution',{})),
            'result_validity':'validated' if valid else 'requires_attention','issues':issues+deepcopy(result.get('collection',{}).get('issues',[])),
            'quality_assessment':{'state':'not_assessed','thresholds':None},
            'coverage':{'expected_frames':result.get('coverage',{}).get('expected_frames'),'actual_frames':count,
            'scenes':len(scene_counts),'scene_frames':dict(scene_counts),'tensor_outputs':len(names),
            'tensor_pairs':len(records),'valid_tensor_pairs':len(valid_records),
            'classes':len(class_rows),'classes_with_gt':sum(r['seen']>0 for r in class_rows),
            'classes_without_gt':sum(r['seen']==0 for r in class_rows),
            'frame_state':result.get('coverage',{}).get('state',{}),
            'finite_output_checks':len(finite['checks']) if isinstance(finite.get('checks'),list) else finite.get('checks'),
            'fresh_outputs':fresh},
            'configuration':{'request':deepcopy(request),'executed':deepcopy(live),
                             'script_config':deepcopy(prov.get('script_config',{}))},
            'tensor_alignment':{'mean_frame_mean_cosine':cosine['mean_cosine_all'],
            'min_frame_mean_cosine':cosine['min_cosine_all'],'min_single_output_cosine':min(r['cosine'] for r in valid_records),
            'outputs':outputs,'frames':frame_rows,'records':records},
            'ground_truth':{'state':'validated' if backend_metrics else 'not_available','summary':gt_summary,
            'classes':class_rows,'counts':count_rows,'receipt':deepcopy(adapt),
            'drivable_counts':{b:d.get('drivable_counts') for b,d in backend_metrics.items()}},
            'findings':{'lowest_cosine':sorted(valid_records,key=lambda r:r['cosine'])[:5],
            'worst_frame_mean':min(frame_rows,key=lambda r:r['mean_cosine']),'supplementary':supplements},
            'source_normalized_result':deepcopy(result),'sources':sources,'field_coverage':field_coverage,
            'sections':[{'id':k,'title':v} for k,v in TITLES],
            'generation':{'mode':'existing_results_only','new_inference_frames':0}}


def fmt(value, digits=4):
    if value is None:return '—'
    if type(value)==int:return f'{value:,}'
    if type(value)==float:return f'{value:.{digits}f}'
    return str(value)


def p(value):return '<p>'+escape(str(value))+'</p>'


def table(headers,rows,widths=None):
    cols='<colgroup>'+''.join(f'<col width="{w}"/>' for w in widths)+'</colgroup>' if widths else ''
    return '<table>'+cols+'<thead><tr>'+''.join('<th background-color="light-gray">'+p(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+p(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'


def render_sections(report, attachments=None):
    """Return independently addressable native-XML sections for publication."""
    c=report['coverage'];t=report['tensor_alignment'];gt=report['ground_truth'];ident=report['identity']
    req=report['configuration']['request'];live=report['configuration']['executed'];result=report['source_normalized_result']
    sections=[]
    def add(key,body):
        title=dict(TITLES)[key]
        sections.append({'id':key,'title':title,'xml':f'<h1 seq="auto">{title}</h1>\n'+body})
    def h2(title):return '<h2 seq="auto">'+escape(title)+'</h2>'
    def bullets(lines):return '<ul>'+''.join('<li>'+escape(str(s))+'</li>' for s in lines)+'</ul>'
    overview=table(['项目','本次结果'],[
        ['评测对象',ident['version_name']],['运行ID',ident['run_id']],
        ['执行状态',{'succeeded':'已完成','failed':'失败'}.get(report['execution'].get('state'),'未确认')],
        ['评测范围',f"{c['scenes']} 个场景 / {c['actual_frames']} 帧；{c['tensor_outputs']} 个输出 / {c['tensor_pairs']} 组比较"],
        ['结果有效性','来源、有限值和范围校验通过' if report['result_validity']=='validated' else '需关注，见有效性检查'],
        ['质量判定','未配置通过阈值，未作通过/不通过判定'],
        ['报告生成','使用已有结果重新组织展示；本次新增推理 0 帧']],[180,580])
    overview+=p('本报告记录一次编译精度评测。张量一致性与相对GT的表现分别统计、分别解释，不计算混合总分。')
    overview+=table(['结果模块','关键观察'],[
        ['张量一致性',f"逐帧平均余弦的均值 {t['mean_frame_mean_cosine']:.6f}；最低单帧单输出余弦 {t['min_single_output_cosine']:.6f}"],
        ['相对GT的表现','；'.join(f"{r['label']}：ONNX {r['onnx']:.2f}% / 板端 {r['board']:.2f}%" for r in gt['summary']) or '未取得GT指标'],
        ['类别覆盖',f"{c['classes_with_gt']} / {c['classes']} 个有效类别有GT，另 {c['classes_without_gt']} 类无GT样本"]],[180,580])
    add('overview',overview)
    model=req.get('model',{});dataset=req.get('dataset',{})
    config=table(['配置项','内容'],[
        ['数据集',dataset.get('class_name','未提供')],['数据版本/划分',Path(dataset.get('data_link_file','未提供')).name],
        ['ONNX',Path(model.get('onnx','未提供')).name],['板端编译产物',Path(model.get('packup_tar','未提供')).name],
        ['推理精度说明',req.get('source',{}).get('reference_preparation','见执行配置')],
        ['实际推理环境',f"Python {live.get('runtime',{}).get('python','未记录')}；ORT provider见执行参数；板端与ONNX共用本次输入"],
        ['输入方式','单路120°相机、冷启动非时序、固定轨迹（已验证的V7参考入口）' if req.get('evaluation',{}).get('v7_pipeline_verified') else '见原始执行配置'],
        ['原推理时间（UTC）',str(ident['inference_started_at'])+' → '+str(ident['inference_finished_at'])],
        ['GT补算时间（UTC）',ident['gt_calculated_at'] or '未补算'],
        ['本报告生成时间（UTC）',report['generated_at']]],[180,580])
    config+=p('模型精度与量化说明以产物和转换配置为准；时序输出参与余弦比较本身不代表完成时序场景评测。完整执行参数与哈希见附件。')
    if req.get('execution',{}).get('adapter')=='subnet_chain_workstation':
        config=table(['配置项','内容'],[
            ['数据集 / 范围','SmartChassisDatasetV7 / 26-10-20-1 / 单帧缓存输入'],
            ['交付源ONNX',Path(model.get('onnx','')).name],
            ['实际FP16参考',Path(live.get('reference_onnx_path','')).name],
            ['源ONNX SHA256',live.get('source_onnx_sha256','未记录')],
            ['FP16参考 SHA256',live.get('onnx_sha256','未记录')],
            ['转换方式','复用原转换器，保留FP8张量；其余浮点转换为FP16'],
            ['板端执行方式','五份原始SubNetOps编译包按接口串接；未合并或重新编译ELF'],
            ['输入与互斥','复用同一帧11路规范输入；使用原Runner原子JSON锁'],
            ['GT指标','本次只验证分网串接及张量对齐，未补算GT'],
            ['实际推理时间（UTC）',str(ident['inference_started_at'])+' → '+str(ident['inference_finished_at'])]], [180,580])
        config+=table(['子网络','原始交付编译包'],[[f"sub{x['stage']}",x['name']] for x in live.get('subnet_archives',[])],[120,640])
    add('configuration',config)
    state=c['frame_state'];cov=table(['检查项','本次范围 / 状态'],[
        ['计划 / 实际帧数',f"{c['expected_frames']} / {c['actual_frames']}"],
        ['帧执行状态',f"完成 {state.get('done','未提供')}；失败 {state.get('failed','未提供')}；待处理 {state.get('pending','未提供')}"],
        ['张量对比有效记录',f"{c['valid_tensor_pairs']} / {c['tensor_pairs']}"],
        ['两端输出有限值检查',f"{c['finite_output_checks']} 份检查，状态见原始校验文件"],
        ['输出来源','已验证每帧推理前清除旧输出' if c['fresh_outputs'] else '来源未通过确认'],
        ['GT与张量样本一致性','同一组帧清单，已核对' if gt['state']=='validated' else '未验证'],
        ['类别支持',f"有GT {c['classes_with_gt']} 类；无GT {c['classes_without_gt']} 类"],
        ['统计层级','GT总体与类别指标覆盖整批；未生成逐场景GT分数']],[230,530])
    cov+=h2('场景与帧清单')+table(['场景','本次帧索引','数量'],[
        [s,', '.join(str(f) for scene,f in live.get('selected_frames',[]) if scene==s),str(n)] for s,n in c['scene_frames'].items()],[500,160,80])
    selection=req.get('evaluation',{}).get('selection',{})
    cov+=p(f"本次实际覆盖 {c['scenes']} 个场景、{c['actual_frames']} 帧。选择配置：{selection or '见原始执行收据'}。覆盖范围不代表全量数据分布。")
    add('coverage',cov)
    tensor=table(['统计量','数值（余弦，无量纲）'],[
        ['逐帧平均余弦的均值',f"{t['mean_frame_mean_cosine']:.6f}"],
        ['最低逐帧平均余弦',f"{t['min_frame_mean_cosine']:.6f}"],
        ['最低单帧单输出余弦',f"{t['min_single_output_cosine']:.6f}"]],[350,410])
    tensor+=p('平均余弦与最低单输出余弦的分母不同。MAE、最大绝对误差使用对应输出的数值单位；不同输出之间不合并求总误差。')
    groups=defaultdict(list)
    for row in t['outputs']:groups[row['group']].append(row)
    for group,rows in groups.items():
        tensor+=h2(group)+table(['输出','有效帧','mean余弦','min余弦','帧MAE均值','最大绝对误差'],[
            [r['label'],str(r['frames']),f"{r['mean_cosine']:.6f}",f"{r['min_cosine']:.6f}",fmt(r['mean_frame_mae']),fmt(r['max_abs_diff'])] for r in rows],[160,70,120,120,140,150])
    tensor+=p(f"余弦保留6位小数展示；完整精度及 {len(t['records'])} 组逐帧比较保存在报告JSON和原始结果附件。极接近1的浮点计算可能略大于1，原始值保持不变。")
    integers=[r for r in t['records'] if r.get('compare_dtype')=='int32' and r.get('integer_exact_ratio') is not None]
    if integers:
        tensor+=h2('整数输出逐元素一致率')+table(['输出','一致率（%）','最大整数差'],[
            [r['onnx_output_name'],fmt(r['integer_exact_ratio']*100,3),fmt(r['max_abs_diff'])] for r in integers],[430,180,150])
        tensor+=p('整数输出同时展示精确一致率，避免仅凭余弦接近1判断逐元素完全一致。')
    add('tensor',tensor)
    ground=h2('总体指标')+table(['指标','ONNX（%）','板端（%）','有效帧'],[
        [r['label'],fmt(r['onnx'],2),fmt(r['board'],2),str(r['frames'])] for r in gt['summary']],[290,170,170,100])
    ground+=p('两端final_occ均由交付脚本的参考CPU聚合得到。以上是相对GT的指标，不是张量相似度；未配置质量门槛。')
    present=[r for r in gt['classes'] if r['seen']>0];absent=[r for r in gt['classes'] if r['seen']==0]
    ground+=h2('有GT样本的类别')+table(['ID / 类别','GT体素数','ONNX IoU（%）','板端 IoU（%）'],[
        [f"{r['class_id']} / {r['name']}",fmt(r['seen']),fmt(r['onnx_iou'],2),fmt(r['board_iou'],2)] for r in present],[270,150,180,180])
    ground+=p('支持数以体素为单位，不是帧数；小类别结果需要结合该分母阅读。present mIoU仅平均上述GT实际出现的类别。')
    ground+=h2('无GT样本的类别覆盖')+table(['ID / 类别','GT体素数','ONNX预测数','板端预测数','状态'],[
        [f"{r['class_id']} / {r['name']}",'0',fmt(r['onnx_positive']),fmt(r['board_positive']),
         '无GT；存在预测' if r['has_predictions'] else '无GT样本'] for r in absent],[260,100,130,130,170])
    ground+=p('无GT类不填造0%或100%，不纳入present mIoU。若有预测仍保留预测数，避免隐藏误检。忽略类别0/15和空类26的规则见指标口径。')
    ground+=h2('逐类支持与交集计数')
    for backend,label in [('onnx','ONNX'),('board','板端')]:
        rows=[r for r in gt['counts'] if r['backend']==backend and (r['seen']>0 or r['positive']>0)]
        ground+=p(label+'：GT/预测/TP为源统计；FP=预测−TP，FN=GT−TP。')
        ground+=table(['类别','GT','预测','TP','FP','FN'],[
            [r['name'],fmt(r['seen']),fmt(r['positive']),fmt(r['correct']),fmt(r['fp']),fmt(r['fn'])] for r in rows],[230,120,120,120,120,120])
    ground+=p(f"其余无GT且无预测的类别计数均为0，完整 {len(gt['classes'])} 类×两端计数保存在报告JSON。")
    ds=gt.get('drivable_counts',{})
    if ds:
        ground+=h2('可行驶区域计数')+table(['后端','GT支持','交集','预测支持'],[
            [b,fmt(v['seen']),fmt(v['correct']),fmt(v['positive'])] for b,v in ds.items()],[180,190,190,190])
    add('gt',ground if gt['state']=='validated' else p('本次没有经过范围与来源验证的GT结果；本模块未评测，不生成占位分数。'))
    findings=p(f"以下按单帧单输出余弦从低到高选取 {len(report['findings']['lowest_cosine'])} 条记录，属于关注样本排序；没有配置通过阈值，不标记为失败。")
    aliases={r['name']:r['label'] for r in t['outputs']}
    findings+=table(['帧编号','输出','余弦','MAE','最大绝对误差'],[
        [f"F{r['frame_ordinal']:03d}",aliases[r['onnx_output_name']],f"{r['cosine']:.6f}",fmt(r['mean_abs_diff']),fmt(r['max_abs_diff'])] for r in report['findings']['lowest_cosine']],[100,200,140,140,170])
    findings+=p('F编号对应本报告场景帧清单的顺序；完整frame_id保留在JSON明细中。')
    worst=report['findings']['worst_frame_mean']
    findings+=p(f"最低逐帧平均余弦：F{worst['ordinal']:03d}，{worst['mean_cosine']:.6f}；样本 {worst['frame_id']}。")
    for extra in report['findings']['supplementary']:
        data=extra['data']
        if extra['name']=='postprocess_consistency.json':
            findings+=h2(f"参考后处理输出差异（仅{extra['scope_frames']}个选定样本）")
            findings+=p('选取三个不同输出的低余弦样本，比较留存的两端标签网格；这是输出一致性诊断，没有使用GT，不是准确率或全批次差异率。')
            findings+=table(['样本','标签不同体素','全网格差异（%）','任一端非空集合差异（%）'],[
                [r['frame_id'],fmt(r['different_label_voxels']),fmt(r['different_label_fraction_all_voxels']*100,2),fmt(r['different_label_fraction_union_nonempty']*100,2)] for r in data['frames']],[330,140,150,200])
        elif extra['name']=='tensor_difference_analysis.json':
            findings+=h2(f"张量诊断（仅{extra['scope_frames']}个选定样本）")
            findings+=table(['输出','样本','余弦','相对L2','最大值索引变化'],[
                [aliases.get(r['output'],r['output']),r['frame_id'],f"{r['cosine']:.6f}",fmt(r['relative_l2']),
                 str(r.get('last_axis_argmax',{}).get('different','—'))+'/'+str(r.get('last_axis_argmax',{}).get('count','—'))] for r in data['checks']],[140,300,120,100,160])
            findings+=p('最大值索引变化与相对L2均为张量诊断，不构成GT精度结论；ONNX节点链仅用于定位输出，当前未确定编译差异根因。')
    if report['issues']:findings+=bullets([str(i) for i in report['issues']])
    add('findings',findings)
    definitions=table(['指标/规则','定义与适用范围'],[
        ['余弦统计','沿用交付CosineAccumulator；输出mean/min分别为该输出在有效帧上的余弦均值/最小值。全局mean为逐帧平均余弦的均值。'],
        ['帧MAE均值','对同一输出的每帧mean_abs_diff做算术平均；不声称是按元素加权的全局MAE。'],
        ['最大绝对误差','同一输出各帧max_abs_diff的最大值；单位随输出数值定义。'],
        ['逐类IoU','先累计本批次交集TP、GT支持和预测支持，再计算TP/(GT+预测−TP)。百分数展示。'],
        ['present mIoU','对有效类别中GT支持数>0的类别IoU取平均；排除无GT类别、忽略类及空类。'],
        ['非空IoU','将有效GT掩码内非空体素视为正类，与语义分类IoU分开累计。'],
        ['可行驶区域IoU','固定源码MeanIoU.single_layer_ds_count(probs=None)分支；使用GT第五列与22个路面/路面设施类别。交集选首个预测可行驶Z层，预测支持累计所有可行驶预测体素。适用于本次未留存概率体的final_occ。'],
        ['GT几何适配','源网格800×48×200、原点(5,-3.6,-2)m；分辨率(0.05,0.15,0.02)m；裁到400×48×100、原点(5,-3.6,-1)m，Z索引减50。'],
        ['类别规则','0(ignore)和15(road_noise)从语义统计中排除；空类26保留用于非空统计，不纳入present mIoU。'],
        ['状态','未取得=来源缺失；无GT样本=GT支持为0；无效=校验未通过；不适用/未启用另行标注。'],
        ['派生字段','帧MAE均值、最大误差汇总、FP/FN和关注样本排序由已有结果计算，没有新增推理。']],[210,600])
    if gt.get('receipt'):
        frames=gt['receipt']['frames']
        checks=[f['geometry_check']['within_one_z_voxel_fraction'] for f in frames
                if isinstance(f.get('geometry_check'),dict) and f['geometry_check'].get('within_one_z_voxel_fraction') is not None]
        if checks:
            prefix='各帧' if len(checks)==len(frames) else f'已提供交叉核验的{len(checks)}/{len(frames)}帧'
            definitions+=p(f"物理坐标交叉核验：{prefix}height_field点落在源OCC相同或相邻Z体素的比例最低为 {min(checks)*100:.4f}%。")
        else:
            definitions+=p('本次采用已验证的V161体素网格到模型ROI适配规则；收据未包含逐帧height_field交叉核验比例。')
    definitions+=p('固定指标参考源码提交：'+str((gt.get('receipt') or {}).get('geometry',{}).get('metric_source_commit','未提供'))+'。本次仅有小场景结果，不设置发版结论。')
    add('definitions',definitions)
    artifacts=h2('输出名称与数值类型映射')+table(['展示名称','实际输出名称','比较dtype'],[
        [r['label'],r['name'],', '.join(r['compare_dtypes'])] for r in t['outputs']],[160,490,130])
    shapes={r['onnx_output_name']:r.get('shape_onnx') for r in t['records'] if r.get('shape_onnx')}
    if len(shapes)==len(t['outputs']):
        artifacts+=h2('实测输出shape')+table(['输出','shape'],[[name,str(shape)] for name,shape in shapes.items()],[520,240])
    else:artifacts+=p('部分原始比较JSON没有记录输出shape；未记录的shape不作推测。')
    artifacts+=h2('模型、数据与样本身份')+table(['身份项','SHA256'],[
        [k,str(ident.get(k) or '未提供')] for k in ['onnx_sha256','packup_sha256','data_link_sha256','selected_frames_sha256']],[220,600])
    artifacts+=h2('信息覆盖与原始产物')+table(['来源','处理情况'],[
        ['统一结果',f"{len(result['metrics'])} 个数值指标完整保留；主要指标及计数在对应模块展示"],
        ['逐帧比较',f"{len(t['records'])} 组记录完整解析并保留；按输出汇总，正文展示最低余弦5例"],
        ['类别支持',f"{len(gt['counts'])} 行两端类别计数保留；无GT类别单列覆盖清单"],
        ['附加诊断',f"{len(report['findings']['supplementary'])} 份附加诊断保留各自样本范围；更多原始字段随原件打包保存"],
        ['尚未取得','业务质量阈值与差异根因未取得；GT未计算项见GT模块。报告生成没有新增推理']],[220,580])
    artifacts+=p('结构化报告与正文使用同一组数据。原始结果包包含不可改写的输入JSON及文件哈希；后续工具可直接读取结构化报告，不需要从正文猜测字段。')
    for attachment in attachments or []:
        artifacts+='<figure view-type="Card"><source path="'+escape(attachment['path'],quote=True)+'" name="'+escape(attachment['name'],quote=True)+'"/></figure>'
    add('artifacts',artifacts)
    return sections


def write_compile_report(output, *, result_file, request_file=None, source_dir=None, supplemental_dir=None,layout='business'):
    if layout not in ('business','diagnostic'):raise ValueError('Unknown report layout')
    output=Path(output).resolve(); result_file=Path(result_file).resolve()
    if output.suffix!='.xml' or output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Use a new XML output path and JSON sidecar')
    result=load_json(result_file);root=Path(source_dir or result['run_dir']).resolve()
    from .validation import require_result
    require_result(result)
    files={'normalized_result.json':{'path':str(result_file),'sha256':digest(result_file)}}
    raw={'normalized_result.json':result_file}
    expected={Path(a['path']).name:a['sha256'] for a in result.get('artifacts',[])}
    for name in ['cosine_summary.json','state.json','execution_receipt.json','input_validation.json','output_integrity.json','tensor_summary_receipt.json']:
        path=root/name
        if path.is_file():
            if name in expected and digest(path)!=expected[name]:raise ValueError('Result artifact content changed: '+name)
            files[name]={'path':str(path),'sha256':digest(path)};raw[name]=path
    cosine=load_json(root/'cosine_summary.json');backends={}
    receipt=result.get('provenance',{}).get('gt_adaptation')
    if receipt:
        for backend in ['onnx','board']:
            name=f'metrics_{backend}.json';path=result_file.parent/name
            if digest(path)!=receipt.get(backend+'_metrics_sha256'):raise ValueError('GT result content changed')
            backends[backend]=load_json(path);files[name]={'path':str(path),'sha256':digest(path)};raw[name]=path
        for name in ['gt_receipt.json','crosscheck.json']:
            path=result_file.parent/name
            if path.is_file():raw[name]=path
    supplementary={}
    if supplemental_dir:
        for name in ['tensor_difference_analysis.json','postprocess_consistency.json']:
            path=Path(supplemental_dir)/name
            if path.is_file():
                supplementary[name]=load_json(path);raw[name]=path;files[name]={'path':str(path.resolve()),'sha256':digest(path)}
    request=load_json(request_file) if request_file else load_json(root/'request.json') if (root/'request.json').exists() else {}
    if request_file:raw['request.json']=Path(request_file).resolve()
    report=build_compile_report(result,cosine,backend_metrics=backends,request=request,source_files=files,supplementary=supplementary)
    report['generation']['layout']=layout
    report['sources'].extend(source_record(n,load_json(pth),files) for n,pth in raw.items() if n not in {s['name'] for s in report['sources']})
    detail=output.with_suffix('.details.zip');output.parent.mkdir(parents=True,exist_ok=True)
    if detail.exists():raise ValueError('Source detail bundle already exists')
    with zipfile.ZipFile(detail,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        manifest=[]
        for name,path in raw.items():
            archive.write(path,'sources/'+name);manifest.append({'name':name,'sha256':digest(path),'bytes':path.stat().st_size})
        import json
        archive.writestr('source-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    report['detail_bundle']={'name':detail.name,'sha256':digest(detail),'files':manifest}
    save_json(output.with_suffix('.json'),report)
    relative=lambda path:'@./'+path.relative_to(output.parent).as_posix()
    sections=render_sections(report,[{'path':relative(output.with_suffix('.json')),'name':'单次编译评测结构化报告.json'},
                                     {'path':relative(detail),'name':'原始结果与完整明细.zip'}])
    if layout=='business':
        from .business_layout import compile_business_sections
        sections=compile_business_sections(report,sections)
    sectiondir=output.parent/(output.stem+'.sections');sectiondir.mkdir(exist_ok=False)
    index=[]
    for section in sections:
        path=sectiondir/(section['id']+'.xml');path.write_text(section['xml'],encoding='utf-8')
        index.append({'id':section['id'],'title':section['title'],'path':str(path),'sha256':digest(path)})
    title='编译精度单次评测报告 '+report['identity']['model_version']
    output.write_text('<title>'+escape(title)+'</title>\n'+'\n'.join(s['xml'] for s in sections),encoding='utf-8')
    save_json(sectiondir/'index.json',{'schema':'roadmodel.report-sections/v1','sections':index,
                                     'report_sha256':digest(output.with_suffix('.json'))})
    return {'xml':str(output),'json':str(output.with_suffix('.json')),'details':str(detail),'sections':str(sectiondir/'index.json'),
            'schema':report['schema'],'frames':report['coverage']['actual_frames'],'new_inference_frames':0}
