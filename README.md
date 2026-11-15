# RoadModel · 模型与编译精度评测

[实习项目总览](https://github.com/Inner-vic/internship-projects) · [使用与验证](docs/project-guide.md)

让模型精度与编译精度共用明确输入，分别执行、核验和报告。

## 解决的问题

同一个模型交付里，权重、配置、ONNX、板端编译产物和评测数据必须保持一致。模型精度与编译精度又有不同的执行环境和成功条件，混在一个“大任务成功”里会掩盖缺失数据或局部失败。

项目用同一份明确输入建立两条独立证据链，把材料解析、执行范围、恢复状态和报告内容绑定起来。

## 实现了什么

| 能力 | 实现结果 |
| --- | --- |
| 两链接提交 | 从模型交付和数据集链接解析材料、版本、索引与 GT |
| 模型精度 | 在已适配的 LPAI 环境执行推理，收集 OCC、视觉和轨迹相关结果 |
| 编译精度 | 工作站调度台架，对比同版 FP16 ONNX 与板端输出，并分别计算相对 GT 的指标 |
| 单次评测报告 | 汇总完成、缺项和失败，生成结构化结果并回填飞书工单 |

## 实现架构

![RoadModel 评测的主要模块、执行顺序和关键边界](https://raw.githubusercontent.com/Inner-vic/internship-projects/main/assets/architecture/roadmodel-evaluation.svg)

`Runtime` 使用 SQLite 保存任务和链路状态。输入先解析为执行计划，再交给相应适配器；原始结果、覆盖范围和发布回读分别留存。

## 关键工程设计

### 1. 冻结可比较的输入

计划绑定模型版本、请求文件摘要、数据范围与逐帧 GT。候选材料有歧义时阻止对应链路，缺 GT 时保留缺项；不从历史材料里悄悄补成一个看似完整的实验。

### 2. 两条链路各自收敛状态

模型计算与台架编译比较分别保存进度、结果和异常。张量一致性与 GT 表现分别报告，局部完成仍能提供明确的覆盖信息，避免用单个总分掩盖差异。

### 3. 复用结果必须重新验证身份

留存结果要核对任务、版本、范围和内容摘要。检查点用于接续已知工作；执行结果未知时先对账，发布或通知失败不自动重新计算。

## 实现与验证范围

历史实机验证覆盖已适配版本的小场景，不能扩展为任意架构、数据格式或全量评测均已支持。完整测试仍有原快照已有失败，涉及外部运行模块、现场证据和挂载契约；范围见[公开验证说明](https://github.com/Inner-vic/roadmodel-evaluation/blob/main/PUBLICATION.md)。

本项目实现单次评测；跨版本比较及发版通过判定由上层流程负责。报告结构见[指标与填报规则](docs/reference-report-policy.md)。

## 源码导读

| 源码 | 关注的实现 |
| --- | --- |
| [pcm_eval/runtime.py](pcm_eval/runtime.py) | 持久任务与双链路推进 |
| [pcm_eval/planning.py](pcm_eval/planning.py) | 计划构建与执行约束 |
| [pcm_eval/retained.py](pcm_eval/retained.py) | 留存结果的身份核验与复用 |
| [pcm_eval/two_link.py](pcm_eval/two_link.py) | 提交解析与材料绑定 |

## 项目资料

使用步骤、依赖和本地验证见[使用与验证](docs/project-guide.md)；源码来源与验证范围见[发布说明](https://github.com/Inner-vic/roadmodel-evaluation/blob/main/PUBLICATION.md)。本页用于介绍实习阶段的实现与设计，具体结论以所链接的代码和证据为准。

[BenchFlow](https://github.com/Inner-vic/benchflow)展示共享硬件编排；[Mesh Visualize](https://github.com/Inner-vic/mesh-visualize)提供独立可视化协议，消费数据需符合其输入要求。
