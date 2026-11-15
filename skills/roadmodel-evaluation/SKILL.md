---
name: roadmodel-evaluation
description: 执行 RoadModel 模型精度或编译精度的输入预检、已有结果收集、独立单次报告生成和飞书交付。用于 PCM 单次评测任务；版本对比属于独立组合能力。
---

定位包含 `pcm_eval/__main__.py` 的 RoadModelEval 项目，以下命令均在项目根目录执行。此 Skill 依赖项目程序，不复制评测实现。

## 统一入口

- 飞书正常业务入口只接受 `model_delivery_url` 和 `dataset_url`，固定同时建立模型精度与编译精度两条子任务；用户无需选择链路或运行环境。见 `README.md`。
- `python -m pcm_eval entry-resolve --inputs examples/two_link_submission.json --output <新预检JSON>` 仅检查旧两链接兼容接口；当前两输入卡片由 EvaluationRuntime 校验，输入示例见 examples/evaluation_submission.json。
- 内部 `roadmodel-model-accuracy`、`roadmodel-compile-alignment` 保持原子，单次报告生成与推理、发布分别记录。以下技术多 job 格式用于内部集成，不要求飞书用户填写。

- 读取项目的 `README.md`，确认本次启用头和材料。
- 已有结果批量生成：`python -m pcm_eval batch-report --submission examples/batch_reports.json --output <新输出目录>`。
- 新评测先解析：`python -m pcm_eval resolve --submission <提交JSON> --output <新解析结果JSON>`。`prepare_execution` 只生成计划。执行单独调用 `execute --plan ...`。
- 输入采用 `roadmodel.submission/v1`，结构见 `schemas/submission.schema.json`。已登记来源可用模型链接或 profile 定位；未知来源返回缺项，不替换成示例版本。每个 job 对应一种评测。

## 报告与交付

两类评测分别生成结果与报告。默认 business 布局沿用原业务表格，每个指标只列本次版本。编译报告保留 ONNX 对板端的本次比较，张量一致性与 GT 指标分开，不生成混合总分。

`batch-report` 不推理或发布，生成各项 JSON/XML、证据包、相对路径交付清单和分阶段收据。用 `inspect-delivery --directory ...` 验证搬迁后的交付。包内历史 source paths 是追溯信息，不能用作新机器的执行路径。

发布飞书时使用项目发布适配器和飞书文档能力，对指定报告做修订绑定更新，严格遵循原模板，不添加诊断或历史执行过程。发布不重新评测。未要求发布时交付本地审阅产物，不自行发送消息。

## 执行边界

沿用用户指定的小场景范围，没有明确要求不扩大为全量、不重新训练。各模型头分别绑定 GT、数据版本和帧清单；无 GT 指标保留不可计算原因。执行、收集、报告、发布和质量判定分别记录。

开发空间迁移前用 `plan-placement` 校验真实挂载、依赖、从空间到台架的路由和板端互斥。PC 能分别 SSH 两端不能证明两端互通。既有专属空间使用官方 lpai-agent 查询和启动，平台要求确认时等待用户确认后继续。未验证项目准确列为待验收。
