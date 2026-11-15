---
name: roadmodel-compile-alignment
description: RoadModel 编译精度原子能力，集成原台架评测脚本，核验同版ONNX与板端产物，生成张量一致性和GT分区的独立单次报告。
---

在 RoadModelEval 项目读取 `README.md`。本能力只执行 Runtime 冻结的 `compile_alignment` 子任务。

复用现有工作站驱动与供应方台架脚本，先核验 ONNX、编译包版本和输入输出接口。原客户端按 model_info.json 执行，不要求 ELF 名为 MergedNet；原单包流程按11路整网输入写入，不能直接拿任意分网包替代。

v3.2.2五分网已通过 `pcm_eval/subnets.py` 的32条边界匹配和 `tools/subnet_chain_remote.py` 的真实单帧验证：按每个包自己的输入顺序串接，尤其sub1是voxel1、voxel0、voxel2；仅明确映射雷达模块名前缀。中间输入必须来自本次上游输出，不能读取包内golden数据。复用原BoardInferClient、InferenceExec、FP16转换器与ORT CPU。

执行范围以 Runtime 冻结的计划和对应执行器契约为准，不把单帧示例绑定作为全量计划。使用原Runner的claim_bench/release_bench原子JSON锁；不得对该锁使用flock或touch。只在已确认没有残留InferenceExec时释放；未知执行保持待核验。版本不匹配时停止，不以旧v3.2.3包替代。

复用已登记工作站数据缓存；仅补选中场景实际缺少的输入及 GT。没有显式授权不跑全量。板端执行、GT计算、结果收集和报告生成保持独立阶段。

已有结果单独 `python -m pcm_eval report-compile --result <结果JSON> --output <新XML路径>`。张量一致性与相对 GT 表现分区，不生成混合总分。执行或发送结果未知时按原收据续收，不重复推理。
