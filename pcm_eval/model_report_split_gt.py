"""Independent renderer for retained split, GT and preprocessing diagnostics."""
from .model_report import h2,p,table

def split_gt_section(a:dict)->str:
    assert a['schema']=='roadmodel.split-gt-diagnostic/v1'
    tr=a['effective_splits']['train'];va=a['effective_splits']['validation'];d=a['domain_shift'];c=a['file_collection']
    out=[h2('训练与验证的水泥覆盖、GT 和预处理核查')]
    out.append(p('本轮补测确认：同一发布权重、验证预处理和原验证帧 20 完全相同的直行 XY 下，一帧训练水泥点 657/657 正确，而原验证水泥帧为 0/680。这更支持场景或条件特征的泛化差异。训练高总体分数不能解读为水泥已普遍学好；目前尚不能将具体主因锁定为类别比例、某项增强或标定变化。'))
    out.append(table(['本轮范围','实际执行'],[
        ['原材料复用','原任务日志、实际配置、固定源码及两帧 logits；复用 2 份既有 GT 文件'],
        ['索引核查',f"读取原任务引用的 {c['manifest_metadata_bytes']:,} 字节数据索引，仅元数据，不含图像/LiDAR/GT 数组"],
        ['小场景 GT 核查','4 个训练场景 + 3 个验证场景，各 1 帧；14 份 Mesh/OCC GT，新增下载约 9.28 MiB'],
        ['计算范围','GT 检查与留存 logits 重计分；另补测训练水泥、沥青各 1 帧，共 2 帧次整网 eval 前向；训练 0 次，未扫描全量 GT'],
        ['环境复用','原授权空间从页面手动启动；复用已校验的权重、源码和算子，未安装运行包；结果已回收并校验哈希']],[180,580]))
    out.append('<h3>实际数据划分与水泥分布证据</h3>')
    out.append(table(['项目','训练','验证'],[
        ['实际合并入口','train + joint_train；train 为空，joint_train 有效','eval + joint_eval 按场景键覆盖合并，未重复累加'],
        ['场景 / 索引帧',f"{tr['scenes']:,} / {tr['listed_frames']:,}",f"{va['scenes']:,} / {va['listed_frames']:,}"],
        ['按原 5 帧、步长 5 规则构造条目',f"{tr['loader_entries']:,}",f"{va['loader_entries']:,}"],
        ['场景交集','两边 0 个场景交集','同一验证索引在 eval / joint_eval 各出现一次'],
        ['GT 声明','所有索引帧均声明 v1-5-1 OCC 和 Mesh 文件','同版本；声明存在不等于逐文件完成校验'],
        ['完整水泥类别占比','索引没有类别字段；原日志/TensorBoard 未保存训练水泥计数','原训练历史验证：2,198,043 / 7,822,213 轨迹点，28.10%']],[220,270,270]))
    out.append(p('训练集水泥的精确占比仍无法从现有索引计算，不能用下方 4 个诊断场景估计 7,771 个场景的总体比例。本轮没有为了补这个比例扫描 619,105 帧 GT。当前索引与 14 份 GT 对象的最后修改时间均早于原任务运行；已保留对象 ETag、时间和内容 SHA256。'))
    out.append('<h3>少量场景的 GT 实查</h3>')
    rows=[]
    for r in a['GT_samples']['frames']:
        q=r['fixed_straight_query'];name=r['scene'].split('/')[-1];label=q['label_counts']
        rows.append(['训练' if r['split']=='train' else '验证',name[:19]+' / '+str(r['frame_index']),f"{r['valid_mesh_cement_fraction']*100:.2f}%",f"{label.get('2',0)}/{q['valid']}",f"{q['OCC_pillar_contains_mesh_label_count']}/{q['valid']}"])
    out.append(table(['划分','采集时间 / 帧','有效 Mesh 水泥比例','直行查询水泥点','Mesh 标签被 OCC 支持'],rows,[70,220,160,150,160]))
    out.append(p('查询统一复用原失败帧的前轮直行 XY；这张表展示 GT 覆盖，模型对照另列下表。4 个训练样本包含一帧有效 Mesh 水泥比例 97.06%、直行查询水泥 657/680 的场景，证明水泥标签 2 确实进入训练数据且未被有效掩码全部丢弃。两帧验证水泥 GT 同样可用。'))
    out.append(p('6 帧的 Mesh 查询标签全部被对应 OCC 柱支持；另一帧为 678/680，差异 2 点出现在沥青/减速带控制样本。Mesh 与 OCC 横向分辨率不同，不能把二者当成逐点完全相同的标注。所查水泥样本未发现水泥/沥青统一错位。样本按同车辆和覆盖控制选择，非随机代表样本；同日训练候选不存在时用了同车辆的已有日期。'))
    out.append('<h3>预处理和轨迹采样的实际差异</h3>')
    out.append(table(['环节','共同部分 / 差异','目前能说明什么'],[
        ['GT 和坐标','同一 V7 加载器、V151 Mesh 采样、类别 2、ROI (5,-3.6,-1)～(25,3.6,1)、相同有效掩码规则','未发现独立的训练/验证标签重编码分支'],
        ['图像公共预处理','相同 resize=1、固定 crop 起点 (856,1120)、输出 768×1600、RGB 转换及 mean/std','不存在声明的裁剪尺寸或归一化差异'],
        ['图像增强','训练随机水平翻转及颜色扰动；验证不翻转、无颜色扰动','翻转同时更新投影矩阵；属于明确差异，尚无证据表明实现错位'],
        ['外参扰动','训练 AUG_PROB=0；验证也不启用','不能将这次差距归为启用的外参随机扰动'],
        ['轨迹构造','训练转向随机 ±200°，稀有类补偿预算 0.3、远端覆盖概率 0.5（±30°）；验证固定 0°，不传两类补偿','GT 查询位置和语义条件位置分布不同；补偿类 16/17/19/23 不含水泥'],
        ['LiDAR 条件','训练以 0.2 概率对每轮轨迹的 3～8 个连续位置做局部 dropout；验证关闭','正常训练增强差异；不是全段 LiDAR 缺失，也未证明导致水泥错误'],
        ['冻结主干模式','源码 train() 会将已冻结主干重新保持在 eval','未发现因训练循环把冻结主干恢复到 train 的静态代码问题'],
        ['计分','训练为主进程抽样 batch、3 条横向带、全类别；验证为分布式中心点和单类水泥召回','不能把训练 99.60% 与水泥验证 14.45% 当成同一个指标直接相减']],[150,340,270]))
    out.append('<h3>按训练 CE 覆盖重计分：两帧仍然误判</h3>')
    out.append(table(['原验证帧','有效三带表面点','水泥 GT / 正确','语义 CE（离线）','标签准确率'],[
        [r['frame_index'],r['valid_surface_tokens'],f"{r['GT_cement_tokens']} / {r['correct']}",f"{r['mean_CE']:.4f}",f"{r['label_accuracy_percent']:.2f}%"] for r in a['aligned_existing_logits']['frames']],[100,150,170,170,170]))
    out.append(p('使用前一轮保存的全类别 FP32 logits、原有效掩码及原 Mesh GT，按确定性 CE 公式重新计算；三条采样带均为水泥，类别权重为 1。这项重计分没有重新推理，也不是原任务全验证集 CE。它说明当前两帧的 0% 不能仅由“中心点与横向带统计口径不同”解释。'))
    out.append('<h3>统一模型与验证预处理的两帧对照</h3>')
    control=a['model_control'];training=control['frames'][0];asphalt=control['frames'][1];prior=control['retained_validation_frames']
    out.append(table(['样本','目标类别正确 / GT','该类中心点 CE','全帧正确 / 有效'],[
        ['训练水泥，4 月 23 日 / 58 帧','657 / 657（100.00%）',f"{training['class_mean_CE']['2']:.6f}",'657 / 680（96.62%）'],
        ['训练沥青，5 月 21 日 / 58 帧','680 / 680（100.00%）',f"{asphalt['class_mean_CE']['1']:.6f}",'680 / 680（100.00%）'],
        ['原验证水泥，7 月 7 日 / 20 帧','0 / 680（0.00%）',f"{prior[0]['cement_mean_CE']:.6f}",'0 / 680（0.00%）'],
        ['原验证水泥，7 月 7 日 / 21 帧','0 / 680（0.00%）',f"{prior[1]['cement_mean_CE']:.6f}",'0 / 680（0.00%）']],[280,190,130,160]))
    out.append(p('新增前向仅为两帧训练来源样本，使用 model.eval、FP32、验证裁剪/归一化，关闭训练图像增强和 LiDAR dropout。权重与模型配置一致，两帧 XY 与原验证帧 20 逐元素相同；帧 21 复用原来的直行 XY，未声称与帧 20 完全相同。输入根目录不同，预处理变换参数一致。'))
    out.append(p('训练水泥帧另有 23 个伸缩缝 GT 点，全部误判成水泥，所以全帧不是 100%。上述结果证明该权重在某个训练水泥场景上能够识别水泥，且同一验证预处理不会令所有水泥失效；仍不能用一帧代表全训练集，或直接宣称已经证明过拟合。'))
    for photo in a['example_images']:
        out.append('<img path="@./'+photo['path']+'" width="760" caption="'+photo['caption']+'"/>')
    out.append(p('原图用于查看同车在不同日期、道路宽度、阴影及周边结构下的场景差异；材质计分仍采用原始 GT，未重新人工标注。'))
    out.append('<h3>数据域差异与当前归因</h3>')
    out.append(p(f"训练来自 {d['training_vehicle_count']} 辆车，覆盖 2026 年 4～8 月；验证来自 {d['validation_vehicle_count']} 辆车，135/138 个场景采于 7 月。验证中有 {d['validation_unseen_vehicle_scenes']}/138 个场景、{d['validation_unseen_vehicle_frames']:,}/{va['listed_frames']:,} 个索引帧来自训练未出现的车辆，分别占 {d['validation_unseen_vehicle_scene_share_percent']:.2f}% / {d['validation_unseen_vehicle_frame_share_percent']:.2f}%。这些是索引场景/帧占比，不是水泥点占比。"))
    out.append(p('原失败水泥帧的车辆在训练里已有 2,507 个场景，且本轮该车辆的训练水泥帧已能正确识别，因此“验证用了训练未见车辆”不能作为原两帧失败的单一解释。'))
    cal=control['calibration_summary']
    out.append(p(f"同车训练/验证水泥场景的相机内参一致，但所附外参不同：相机相对旋转约 {cal['camera_extrinsic_rotation_difference_deg']:.3f}°，LiDAR 相对旋转约 {cal['lidar_extrinsic_rotation_difference_deg']:.3f}°、平移约 {cal['lidar_translation_difference_cm']:.2f} cm。模型按各自场景加载标定；这些是元数据差异，不等于标定错误。本轮没有交换标定的因果试验。"))
    out.append(p('当前最有支持的方向是水泥场景或条件特征的泛化差异：固定验证预处理和相同直行 XY 后，训练水泥帧与验证水泥帧仍分化，三带 CE 对齐也未消除验证错误。具体归因仍需区分场景外观、标定与传感器条件、训练水泥有效采样比例，以及轨迹/增强覆盖；现有小样本不能单独确定其中一项为根因。'))
    out.append(p('本轮保留原报告的小场景评测结果，并将新增两帧作为独立诊断范围。评测、GT/指标分析、报告生成和飞书发布分开调用，继续保持独立原子能力；未重新训练或跑全量。'))
    return ''.join(out)
