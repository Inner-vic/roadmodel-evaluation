# 使用与验证

[返回项目介绍](../README.md)

本页保留具体使用、环境和验证入口；命令均从仓库根目录执行，示例路径需按自己的环境替换。

面向 RoadModel 的单次评测工具：以模型交付链接和数据集链接为输入，核对版本与材料，分别执行模型精度、编译精度评测，再生成可追溯的飞书报告。

公开仓库提供编排、适配、测试与报告源码。真实执行需要已授权的 LPAI 计算环境、模型与数据、工作站/台架，以及飞书接入配置；安装 Python 包后仍需完成这些环境准备。

## 两条评测链路

| 链路 | 执行位置与内容 | 结果 |
| --- | --- | --- |
| 模型精度 | LPAI 中读取数据与 GT，运行已适配模型 | OCC、视觉分类、轨迹等已适配评测项及实际覆盖范围 |
| 编译精度 | 工作站调度台架，对比同版 FP16 ONNX 与板端输出 | 张量一致性，以及两端相对 GT 的独立指标 |

两条链路分别记录状态、缺项与失败。张量一致性和 GT 表现分别呈现；缺少可靠 GT 时注明不可计算，不以零值或其他版本数据补齐。

## 输入与交付

每次提交两项业务输入：

- **模型交付链接**：关联权重、配置、同版 ONNX 和编译产物。
- **评测数据集链接**：明确数据版本、索引和逐帧 GT。

系统先解析并冻结材料，再准备环境、执行评测、收集结果，最终回填原卡。部署后的飞书入口为 `RoadModel评测`；字段见[提交示例](../examples/evaluation_submission.json)与[输入结构](../schemas/evaluation_inputs.schema.json)。

交付包括 PCM 单次评测报告、`pcm.single-eval/v1` 结构化结果，以及输入绑定、检查点和原始结果的核验记录。报告失败可续交付，不据此重复推理或台架执行。

## 本地安装

使用 Python 3.10+，建议在独立虚拟环境中执行：

```powershell
git clone https://github.com/Inner-vic/roadmodel-evaluation.git
cd roadmodel-evaluation
python -m pip install -e .
python -m pcm_eval --help
```

按需安装 GT 与视觉处理依赖：

```powershell
python -m pip install -e ".[gt,vision]"
```

模型运行镜像、供应方源码、台架原生工具和飞书/LPAI 客户端另行准备。当前模型适配器使用 CUDA GPU，尚未适配 CPU 推理。

## 部署接入

1. 准备本人可访问的 LPAI 资源、官方 SDK/客户端和已有 SSH 凭据引用。
2. 运行配置引导，核对参考空间、实际账号、资源和镜像：

   ```powershell
   python -m pcm_eval configure-environment --output initialization-request.json
   ```

3. 按审核后的配置初始化并保存绑定状态：

   ```powershell
   python -m pcm_eval initialize-environment --request initialization-request.json --state-dir .runtime/installation --output evidence/initialization
   ```

4. 将初始化结果接入自己的桌面 Agent、服务端配置和飞书工单入口，核对数据挂载与台架队列，再按明确范围完成小场景验收。

配置引导及初始化参数见各命令的 `--help`，请求结构见[初始化示例](../examples/installation_request.example.json)。`BOUND` 只表示空间绑定已核验，机器人接入和双链路执行还需分别验证。平台创建、启停与台架操作遵循各自的权限及确认要求。

`examples/`、`configs/` 中保留了历史契约和脱敏参考值；其中的链接、路径与资源标识不能直接当作新环境的可用配置。

## 源码结构

| 路径 | 内容 |
| --- | --- |
| `pcm_eval/` | 输入解析、任务状态、调度、收集与报告生成 |
| `skills/` | 模型精度、编译精度、报告和开发空间相关 Skill |
| `tools/` | 运行适配、后台 Agent 与部署工具 |
| `schemas/`、`examples/`、`configs/` | 数据结构、输入及配置参考 |
| `tests/` | 单元测试、契约和环境相关集成测试 |

## 验证与支持范围

源码检查和不依赖真实材料的调度测试：

```powershell
python -B tools/check_public_source.py
python -B -m unittest tests.test_batch_schedule -v
```

GitHub 的 [Source checks](../.github/workflows/source-checks.yml) 只执行源码检查。完整测试套件仍有原快照已有失败，涉及未入库的现场证据、外部运行模块及已有挂载契约；具体范围见[发布说明](../PUBLICATION.md)。

历史实机验收仅覆盖已适配版本的小场景，不能推断任意模型、任意数据格式或全量评测均已支持。跨版本比较和发版通过判定由上层流程负责。

报告结构与指标映射见[报告填报规则](../docs/reference-report-policy.md)和[模板配置](../configs/report_template.json)。模型、数据集、凭据及运行数据库不随源码分发。
