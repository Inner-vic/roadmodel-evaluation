"""Render an existing training audit as an independent model-report section."""
from .model_report import p,h2,table

def training_semantic_section(a:dict)->str:
    assert a['schema']=='roadmodel.training-semantic-audit/v1'
    epochs=a['epochs'];release=epochs[a['checkpoint']['epoch']-1]
    c=release['classes'];tr=release['sampled_training_scalars'];find=a['findings']
    out=[h2('原训练任务的语义表现核查'),p('已确认水泥识别弱项在原训练任务的历史验证中就已存在，当前两帧的 0% 与该弱项方向一致。历史验证仍能识别部分水泥，不能称为全模型恒定输出沥青。具体训练成因尚未闭环：现有材料缺少训练集水泥直方图、分水泥 CE 和完整混淆矩阵。')]
    out.append(table(['证据范围','本轮核查'],[
        ['原训练任务',a['task_uid']],
        ['原任务输出','test-gzw / 26-09-07-e7nn；已核对资产来源标签与任务 UID'],
        ['发布权重','epoch_44.pth；内嵌 epoch=44、global_iter=23012=44×523，对应日志 log_epoch=43'],
        ['材料完整性','原任务日志、实际配置、TensorBoard；60 轮日志与标量交叉一致；176611 条事件记录 CRC 已验证'],
        ['执行范围','仅解析历史材料，新增训练 0 次、推理 0 帧；本报告原有小场景指标保持原值']],[180,580]))
    out.append('<h3>水泥覆盖与分类表现</h3>')
    out.append(table(['类别','历史验证有效轨迹点','点数占比','第 44 轮召回率'],[
        ['沥青（1）',f"{c['asphalt']['support']:,}",f"{release['validation_class_share_percent']['asphalt']:.2f}%",f"{c['asphalt']['tensorboard_recall_percent']:.2f}%"],
        ['水泥（2）',f"{c['cement']['support']:,}",f"{release['validation_class_share_percent']['cement']:.2f}%",f"{c['cement']['tensorboard_recall_percent']:.2f}%"],
        ['其他有 GT 类别',f"{release['validation_valid_points']-c['asphalt']['support']-c['cement']['support']:,}",f"{100-release['validation_class_share_percent']['asphalt']-release['validation_class_share_percent']['cement']:.2f}%",'逐类结果见附件'],
        ['合计',f"{release['validation_valid_points']:,}",'100.00%',f"总体标签准确率 {release['tensorboard_validation_accuracy_percent']:.2f}%"]],[190,190,160,220]))
    out.append(p('上述为原训练任务分布式聚合的验证轨迹点统计，不是独立场景数或去重帧数，也不是训练集类别占比。60 轮水泥支持数相同；不能把 60 轮的点数累加为新增样本。水泥占验证点 28.10%，足以排除“验证中没有水泥”的解释。'))
    out.append(table(['权重轮次','水泥召回率','沥青召回率','总体标签准确率'],[
        [str(r['checkpoint_epoch']),f"{r['classes']['cement']['tensorboard_recall_percent']:.2f}%",f"{r['classes']['asphalt']['tensorboard_recall_percent']:.2f}%",f"{r['tensorboard_validation_accuracy_percent']:.2f}%"] for r in [epochs[i] for i in [0,19,27,43,59]]],[150,200,200,210]))
    out.append(p(f"60 轮水泥召回率范围为 {find['cement_recall_min_percent']:.2f}%～{find['cement_recall_max_percent']:.2f}%，未恢复到与沥青相近的水平。第 44 轮约 {find['cement_share_of_validation_errors_percent']:.2f}% 的标签错误来自水泥 GT 点。该比例由类别支持数与召回率计算；没有完整混淆矩阵，不能断言历史验证中这些错误全部被预测为沥青。"))
    out.append('<h3>语义损失：已记录与未计算的部分</h3>')
    out.append(table(['指标','第 44 轮 / 状态','口径'],[
        ['训练语义 CE',f"{tr['label_ce']['mean']:.5f}",'主进程每 10 步记录的 53 个 batch 标量的简单均值'],
        ['训练语义准确率',f"{tr['label_acc']['mean']*100:.2f}%",'同批次的有效横向表面 token，非水泥专属或全训练集精度'],
        ['历史验证总体准确率',f"{release['tensorboard_validation_accuracy_percent']:.2f}%",'分布式聚合的轨迹中心点；与上行覆盖及采样口径不同'],
        ['验证语义 CE','未计算（N/A）','日志 Validation/Loss=0.000 来自损失函数提前返回，不能作为零语义误差'],
        ['水泥单类 CE / 训练水泥占比','现有产物未记录','原输出 67 个文件及 156 个 TensorBoard tag 未提供对应统计']],[190,210,360]))
    out.append(p('训练 CE 持续为非零值，说明语义监督确实参与训练。训练采样准确率高、验证水泥召回率低，提示需要核查训练/验证数据和特征泛化差异，但这些口径不能直接等同，尚不足以证明过拟合或训练水泥样本不足。'))
    out.append(p('验证 Loss=0 的代码路径：inference() 不输出 surface_pred / surface_target；SurfacePatchDDPMLoss.forward() 在缺少它们时直接返回 0，尚未执行后面的语义 CE。当前报告将该验证损失标为 N/A，保留原值仅作追溯。'))
    out.append('<h3>配置、选模与可确认的问题</h3>')
    out.append(table(['核查项','事实','解释边界'],[
        ['确定性语义 CE','总权重 1.0；水泥与沥青类权重均为 1，减速带（16）为 5','水泥未被配置为忽略类；未设水泥专属加权'],
        ['SEMANTIC_CLASS_BALANCE=False','该开关位于语义扩散损失分支；本任务语义扩散权重为 0','它不控制当前确定性 CE，不能仅凭 False 判定根因'],
        ['稀有类别补偿','16、17、19、23，未包含水泥（2）','未显式补偿水泥，不等于已证明水泥训练样本少'],
        ['训练方式','冻结主干，加载 V151 E83 主干及 geom-profile E30 轨迹头后训练 60 轮','首轮训练后的验证已弱；未留存迁移后训练前的语义验证，无法区分继承偏差与本轮首轮影响'],
        ['自动 best 规则','优先稳定性、高度及前沿事件；总体 label_acc 仅位于末级 tie-break','未设水泥召回门槛，几何选模不会保证水泥语义达标'],
        ['best_traj 与发布轮次','epoch_44 内嵌 best_traj.epoch=27，表示日志第 27 轮、权重第 28 轮','它是截至当时的历史最优元数据，不能当作 epoch_44 的当前指标；发布 E44 的人为选择依据未附说明']],[170,310,280]))
    out.append(p('可确认的问题有三项：水泥分类在原训练验证中长期偏弱；验证语义损失未被实际计算；自动选模缺少水泥单类质量约束。后两项是可观测性和验收设计缺口，不能反向证明它们造成了错误权重。'))
    out.append(p('具体训练成因仍待证据：训练集与验证集按场景/轨迹点的水泥覆盖、分水泥 CE 与混淆矩阵、迁移初始权重的既有验证，以及 E44 发布选择依据。目前不需要重新训练或全量推理来确认上述历史弱项；进一步归因应先补齐这些已有统计或训练归档。'))
    return ''.join(out)
