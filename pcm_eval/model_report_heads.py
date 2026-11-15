"""Presentation of independently scoped head results; no execution or publication."""
from html import escape
from pathlib import Path
from .model_report import p,h2,table

TASK_NAMES={'rainfall_intensity':'降雨强度','basic_road_surface':'基础路面','surface_attachment':'路面附着物',
    'surface_attachment_distribution':'附着物分布','overall_road_bumpiness':'整体颠簸程度',
    'bump_type':'颠簸类型','bridge_joint_type':'桥梁接缝类型','pothole_damage':'坑洞破损'}

def scope_xml(result):
    run=result['provenance']['native_run'];env=run.get('environment',{});selected=run['selected_frames']
    rows=[[scene,', '.join(str(fr) for ss,fr in selected if ss==scene)] for scene in dict.fromkeys(s for s,_ in selected)]
    return p('运行 ID：'+run['run_id'])+table(['项目','本任务输入与执行'],[
        ['数据版本 / 加载器',run.get('dataset_version','')+' / '+run.get('dataset_class','')],
        ['执行模式',f"HEAD_MODE={env.get('HEAD_MODE')}；TRAIN_MODE={env.get('TRAIN_MODE')}"],
        ['样本范围',f"{len(rows)} 个场景 / {len(selected)} 帧"],
        ['原始推理流程耗时',f"{run.get('source_inference_seconds',run.get('duration_seconds',0)):.2f} 秒（含初始化）"],
        ['输入清单',run.get('data_source',{}).get('path','见执行收据')]], [200,560])+table(['场景','帧索引'],rows,[610,150])

def render_vision(r):
    items=r['vision_metrics'];out=[]
    if 'vision' in r.get('head_results',{}):out.append(scope_xml(r['head_results']['vision']))
    tasks={};classes={}
    for m in items:
        d=m.get('dimensions',{});task=d.get('task')
        if not task:continue
        if 'class_id' in d:classes.setdefault((task,d['class_id']),{})[m['name'].removeprefix('vision.class.')]=m['value']
        else:tasks.setdefault(task,{})[m['name'].removeprefix('vision.')]=m['value']
    core={t:v for t,v in tasks.items() if 'accuracy' in v}
    out.extend([p('GT 来自原分类任务的数据链接及帧级 JSON；原生 eval.py 直接执行分类评测。以下核心任务分别计分。'),
        table(['任务','正确 / 总数','Accuracy（%）','mAP','宏平均 Recall（%）'],[
            [TASK_NAMES.get(t,t)+f' ({t})',f"{v.get('correct','—')} / {v.get('total','—')}",v.get('accuracy'),v.get('map'),v.get('macro_recall')]
            for t,v in tasks.items() if 'accuracy' in v],[280,110,130,100,140]),
        p('源码 mAP 仅对支持度满足条件的类别求均值。应结合下方每类支持数、Recall 和上方实际正确数判断；mAP 与 Accuracy 的定义不同。'),
        h2('有 GT 类别明细'),table(['任务','GT 类别 ID','支持数','Recall（%）','AP'],[
            [TASK_NAMES.get(t,t),cid,v.get('support'),v.get('recall'),v.get('ap')] for (t,cid),v in classes.items()],[280,120,100,140,120])])
    if core and all(len([1 for (task,_),v in classes.items() if task==t and v.get('support',0)>0])==1 for t in core):
        out.append(p('覆盖限制：本次每个核心任务都只覆盖 1 个 GT 类别。对应类别标签全为正样本，AP 退化为 1，不能据此认定分类全对。'+
            '；'.join(f"{TASK_NAMES.get(t,t)}实际 {v['correct']}/{v['total']} 正确（{v['accuracy']:.2f}%）" for t,v in core.items() if v.get('accuracy',100)<100)))
    extras=[[t,v['map']] for t,v in tasks.items() if 'accuracy' not in v and 'map' in v]
    if extras:out.extend([h2('原生日志附加任务'),table(['任务','源码 mAP'],extras,[580,180]),p('附加任务的源码 mAP 原样保留。原生日志未提供这些任务的 Accuracy 和逐类计数；不补造数值。')])
    return ''.join(out)

def render_trajectory(r):
    result=r.get('head_results',{}).get('trajectory');out=[]
    if not result:return table(['指标','维度','值','单位'],[[x['name'],str(x.get('dimensions',{})),x['value'],x['unit']] for x in r['trajectory_metrics']])
    if result:out.append(scope_xml(result))
    m={x['name'].removeprefix('trajectory.'):x['value'] for x in r['trajectory_metrics'] if not x.get('dimensions')}
    figure=r.get('trajectory_figure',{})
    out.extend([p('GT 复用原轨迹任务的 Mesh 高度场和语义标签。轨迹由原生车辆参数与轨迹构造函数生成；复用原生轨迹评测统计函数，具体 GT 来源与构造参数见执行配置。'),
        h2('高度与几何表现'),table(['指标','值','单位'],[
            ['有效点数',m.get('valid_points'),'点'],['连续高度 MAE',m.get('z_float_mae_cm'),'cm'],
            ['连续高度 RMSE',m.get('z_float_rmse_cm'),'cm'],['连续高度 Bias（预测−GT）',m.get('z_float_bias_cm'),'cm'],
            ['离散高度准确率',m.get('zidx_acc'),'%'],['离散高度 MAE',m.get('zidx_mae_bins'),'bin'],
            ['离散高度 MAE',m.get('zidx_mae_cm'),'cm'],['一阶平滑比',m.get('smooth_ratio_d1'),'ratio'],
            ['二阶平滑比',m.get('smooth_ratio_d2'),'ratio'],['形状比误差',m.get('shape_ratio_error'),'ratio']], [420,170,170]),
        p('离散高度按源码采用连续预测与 GT 四舍五入到最近 bin 后计分。平滑比及形状比误差为源码派生统计，未设置业务门限。'),
        h2('语义标签表现'),table(['指标','结果'],[['标签准确率（%）',m.get('label_acc')]], [420,340])])
    if figure:
        gt={};pred={}
        for frame in figure['frames']:
            for k,v in frame['gt_label_counts'].items():gt[str(k)]=gt.get(str(k),0)+v
            for k,v in frame['predicted_label_counts'].items():pred[str(k)]=pred.get(str(k),0)+v
        out.append(table(['来源','类别 ID / 名称','点数'],[['GT',f'{k} / '+{'1':'沥青','2':'水泥'}.get(k,'见原始标签表'),v] for k,v in gt.items()]+[['预测',f'{k} / '+{'1':'沥青','2':'水泥'}.get(k,'见原始标签表'),v] for k,v in pred.items()],[180,400,180]))
        if set(gt)=={'2'} and set(pred)=={'1'}:out.append(p('本次有效 GT 均为水泥路面（2），预测均为沥青（1），语义准确率为 0%。这是本次真实模型表现，需后续分析；不能据高度误差较小判定轨迹头整体通过。'))
        out.append(table(['帧索引','左前轮有效点','右前轮有效点','左后轮有效点','右后轮有效点'],[[x['frame']]+x['valid_points_per_wheel'] for x in figure['frames']], [120,160,160,160,160]))
        if all(x['valid_points_per_wheel'][2:]==[0,0] for x in figure['frames']):out.append(p('本次 front-only 轨迹头仅输出左右前轮有效轨迹，后轮被源码显式屏蔽，后轮不计分。'))
    out.extend([
        h2('前沿事件与边界覆盖'),table(['统计项','GT 事件','预测事件','匹配事件'],[['前沿事件数',m.get('front_gt_events'),m.get('front_pred_events'),m.get('front_matched_events')]], [280,160,160,160]),
        p('没有 GT / 预测 / 匹配事件时，相关前沿质量指标按各自分母标为未覆盖，不填 0 分或 100%。边界输出没有可计分分母时也不计分。原始计数及不可用原因保留在附件。')])
    unavailable=result.get('provenance',{}).get('unavailable_metrics',{})
    if unavailable:out.append(table(['未计分指标','原因'],[[k,v] for k,v in unavailable.items()], [480,280]))
    if figure:
        out.extend([h2('高度曲线'),'<img path="@./'+escape(Path(figure['image']).as_posix())+'" width="760" caption="两帧左右前轮高度曲线：黑色为 Mesh GT，蓝色虚线为预测；横轴为前向距离（米），纵轴为 VCS 高度（厘米）。"/>',
            p('曲线直接使用留存预测逐点转换为物理高度，无平滑或插值；各图共用坐标范围。后轮为该模型结构屏蔽，未绘制；原始逐点 CSV、有效掩码及图表转换收据在证据包中。')])
    run=(result or {}).get('provenance',{}).get('native_run',{})
    if run.get('source_inference_exit_code'):
        out.append(p('执行追溯：两帧推理及预测保存均已完成，首次流程在把无事件分位数 NaN 写入严格 JSON 时退出（exit 1）。随后校验留存 NPZ 的哈希，复用同版原生函数重新统计，未新增推理；无定义数值转为 null，并保留原失败收据。'))
    return ''.join(out)
