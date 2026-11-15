from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .bundle import REFERENCE_SHA256, prepare_bundle
from .common import load_json, save_json
from .datalink import inspect_datalink
from .lpai import collect_task
from .parsers import collect_compile, collect_model
from .planning import build_plan
from .runner import collect_remote, execute
from .single_report import write_report
from .compile_report import write_compile_report
from .model_report import write_model_report


def summary(result: dict) -> dict:
    return {"evaluation_type": result["evaluation_type"], "version_name": result["version_name"],
            "execution": result["execution"], "collection": result["collection"],
            "coverage": result["coverage"], "metric_count": len(result["metrics"]),
            "quality_assessment": result["quality_assessment"]}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PCM 单次评测：计划、触发、收集、解析；不做历史版本对比")
    commands = p.add_subparsers(dest="command", required=True)
    from .installation_cli import add_arguments
    add_arguments(commands.add_parser('initialize-environment',help='首次绑定本机账号，按模板创建并固定专属开发空间；不执行评测'))
    from .installation_guide import add_arguments as guide_arguments
    guide_arguments(commands.add_parser('configure-environment',help='引导选择 GPU 环境并生成本账号初始化配置；只读，不创建空间'))
    inspect = commands.add_parser('delivery-materials', help='在线读取交付记录/文件夹，自动区分两条评测链路的材料')
    inspect.add_argument('--url', required=True)
    inspect.add_argument('--output', required=True)
    delivery = commands.add_parser('evaluate-delivery', help='交付链接与一份数据：核验材料，分别执行模型和编译评测')
    delivery.add_argument('--url', required=True)
    delivery.add_argument('--dataset')
    delivery.add_argument('--output', required=True)
    delivery.add_argument('--catalog')
    delivery.add_argument('--prepare-only', action='store_true')
    delivery.add_argument('--validation', action='store_true', help='内部验收：仅运行已明确登记的小样本，不冒充完整验证集')
    resume = commands.add_parser('continue-delivery', help='续收同一冻结工作流，重做报告不会重复推理')
    resume.add_argument('--directory', required=True)
    entry = commands.add_parser('entry-resolve', help='两个链接解析为固定双链路预检；不推理')
    entry.add_argument('--inputs', required=True)
    entry.add_argument('--output', required=True)
    entry.add_argument('--task-id', default='rm-preflight')
    entry.add_argument('--catalog')
    collect = commands.add_parser("collect", help="解析已存在的单次评测结果")
    collect.add_argument("--kind", choices=("model_accuracy", "compile_alignment"), required=True)
    collect.add_argument("--run-dir", required=True)
    collect.add_argument("--version-name")
    collect.add_argument("--expected-frames", type=int)
    collect.add_argument("--log-file")
    collect.add_argument("--scheme-file")
    collect.add_argument("--gt-dir", help="Verified GT-only results for this exact retained compile run")
    collect.add_argument("--output", required=True)
    collect.add_argument("--overwrite", action="store_true")
    lpai = commands.add_parser("collect-lpai", help="通过只读 CLI 收集一个已有 LPAI 评测任务")
    lpai.add_argument("--task-uid", required=True)
    lpai.add_argument("--output-dir", required=True)
    lpai.add_argument("--version-name")
    lpai.add_argument("--pod")
    lpai.add_argument("--limit", type=int, default=5000)
    remote = commands.add_parser("collect-ssh", help="复用 SSH 配置收集工作站/开发空间结果文件")
    remote.add_argument("--ssh-alias", required=True)
    remote.add_argument("--remote-run-dir", required=True)
    remote.add_argument("--kind", choices=("model_accuracy", "compile_alignment"), required=True)
    remote.add_argument("--output-dir", required=True)
    remote.add_argument("--version-name")
    remote.add_argument("--python", default="python3")
    remote.add_argument("--max-file-mb", type=int, default=64)
    plan = commands.add_parser("plan", help="生成可检查的单次执行计划；不启动评测")
    plan.add_argument("--request", required=True)
    plan.add_argument("--run-id")
    plan.add_argument("--output", required=True)
    run = commands.add_parser("execute", help="执行已补齐参数的计划，自动收集原始产物")
    run.add_argument("--plan", required=True)
    run.add_argument("--evidence-dir", required=True)
    run.add_argument("--resume", action="store_true", help="续跑同一冻结批次计划；使用新的本地证据目录")
    run.add_argument("--batch-limit", type=int, help="本次最多新增多少批；保存检查点后返回 75")
    report = commands.add_parser("report", help="从已有结果生成独立单次报告 XML / JSON，供飞书发布")
    report.add_argument("--kind", required=True, choices=("model_accuracy", "compile_alignment"))
    report.add_argument("--result")
    report.add_argument("--template-only", action="store_true")
    report.add_argument("--request-file")
    report.add_argument("--readiness-file")
    report.add_argument("--template-file")
    report.add_argument("--source-dir", help="已有运行的原始比较和执行元数据目录")
    report.add_argument("--supplemental-dir", help="与本次运行绑定的现有诊断JSON目录")
    report.add_argument("--output", required=True)
    compile_report = commands.add_parser('report-compile', help='原子能力：仅从已有结果生成单次编译精度报告，不触发评测或发布')
    compile_report.add_argument('--result', required=True)
    compile_report.add_argument('--request-file')
    compile_report.add_argument('--source-dir')
    compile_report.add_argument('--supplemental-dir')
    compile_report.add_argument('--output', required=True)
    model_report = commands.add_parser('report-model', help='原子能力：仅从已有结果生成模型精度单次报告，不触发评测或发布')
    model_report.add_argument('--result', required=True)
    model_report.add_argument('--binding-file')
    model_report.add_argument('--gt-preparation-file')
    model_report.add_argument('--head-result',action='append',default=[],help='同版本独立样本范围的视觉或轨迹结果，可重复')
    model_report.add_argument('--trajectory-figure-receipt',help='已落盘轨迹图及源文件校验收据')
    model_report.add_argument('--supporting-artifact',action='append',default=[],help='显式附带配置或输入收据，可重复')
    model_report.add_argument('--output', required=True)
    model_report.add_argument('--layout', choices=['business','diagnostic'],default='business')
    compile_report.add_argument('--layout', choices=['business','diagnostic'],default='business')
    resolve=commands.add_parser('resolve',help='统一输入解析与预检；不触发评测或发布')
    resolve.add_argument('--submission',required=True)
    resolve.add_argument('--profiles')
    resolve.add_argument('--output',required=True)
    batch=commands.add_parser('batch-report',help='批量生成独立单次报告及可搬迁交付包；不推理、不发布')
    batch.add_argument('--submission',required=True)
    batch.add_argument('--profiles')
    batch.add_argument('--output',required=True)
    validate=commands.add_parser('validate-result',help='校验结构化结果契约及源文件哈希')
    validate.add_argument('--result',required=True)
    delivery=commands.add_parser('inspect-delivery',help='读取并校验可搬迁报告交付清单')
    delivery.add_argument('--directory',required=True)
    placement=commands.add_parser('plan-placement',help='用实测挂载/网络探测结果检查编译评测迁移，不触发评测')
    placement.add_argument('--request',required=True)
    placement.add_argument('--target',required=True)
    placement.add_argument('--probe',required=True)
    placement.add_argument('--output',required=True)
    bundle = commands.add_parser("prepare-bundle", help="解包并核验用户已提供的离线包；不运行包内脚本")
    bundle.add_argument("--zip", required=True)
    bundle.add_argument("--destination", required=True)
    bundle.add_argument("--expected-sha256", default=REFERENCE_SHA256)
    inventory = commands.add_parser("inspect-datalink", help="读取真实 data_link 的评测划分并建立元数据清单")
    inventory.add_argument("--file", required=True)
    inventory.add_argument("--split", action="append", choices=("eval", "joint_eval", "vision_eval"))
    inventory.add_argument("--output", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    try:
        if args.command == 'configure-environment':
            from .installation_guide import configure
            return configure(args)
        if args.command == 'initialize-environment':
            from .installation_cli import initialize
            return initialize(args)
        if args.command == 'delivery-materials':
            from .delivery_source import LarkSource
            value = LarkSource().read(args.url)
            save_json(args.output, value)
            print(json.dumps({'version': value['version'], 'chains': value['chains'], 'output': args.output}, ensure_ascii=False, indent=2))
            return 0 if all(c['materials_ready'] for c in value['chains'].values()) else 2
        if args.command == 'evaluate-delivery':
            from .delivery_workflow import prepare_delivery, run_delivery
            target = Path(args.output).resolve()
            if target.exists():
                raise ValueError('请使用新运行目录；接续原任务请用continue-delivery')
            value = prepare_delivery(args.url, args.dataset, catalog_file=args.catalog, validation=args.validation)
            save_json(target/'workflow.json', value)
            if args.prepare_only:
                result = {'ready': value['ready'], 'scope': value['scope'], 'jobs': [
                    {k: j[k] for k in ('kind', 'ready', 'issues', 'environment')} for j in value['jobs']]}
            else:
                result = run_delivery(value, target)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result.get('state') == 'SUCCEEDED' or result.get('ready') is True else 2
        if args.command == 'continue-delivery':
            from .delivery_workflow import run_delivery
            target = Path(args.directory).resolve()
            result = run_delivery(load_json(target/'workflow.json'), target)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result['state'] == 'SUCCEEDED' else 2
        if args.command == 'entry-resolve':
            from .two_link import resolve_pair
            value = resolve_pair(load_json(args.inputs), args.task_id, catalog_file=args.catalog)
            save_json(args.output, value)
            print(json.dumps({'ready':value['ready'],'jobs':[{k:x[k] for k in ('kind','ready','issues','heads')} for x in value['jobs']]},ensure_ascii=False,indent=2))
            return 0 if value['ready'] else 2
        if args.command=='resolve':
            from .intake import write_resolution
            value=write_resolution(args.submission,args.output,profile_file=args.profiles)
            print(json.dumps({'ready':value['ready'],'action':value['action'],'output':str(Path(args.output).resolve()),
                'jobs':[{key:job[key] for key in ('id','kind','profile','ready','execution_heads','plan_file') if key in job}
                        for job in value['jobs']],'issues':value['issues']},ensure_ascii=False,indent=2));return 0 if value['ready'] else 2
        if args.command=='batch-report':
            from .workflow import run_report_batch
            value=run_report_batch(load_json(args.submission),args.output,profile_file=args.profiles)
            print(json.dumps(value,ensure_ascii=False,indent=2));return 0 if value['state']=='succeeded' else 2
        if args.command=='validate-result':
            from .validation import validate_result
            value=validate_result(load_json(args.result))
            print(json.dumps(value,ensure_ascii=False,indent=2));return 0 if value['valid'] else 2
        if args.command=='inspect-delivery':
            from .delivery import inspect_delivery
            print(json.dumps(inspect_delivery(args.directory),ensure_ascii=False,indent=2));return 0
        if args.command=='plan-placement':
            from .placement import migration_plan
            value=migration_plan(load_json(args.request),load_json(args.target),load_json(args.probe))
            save_json(args.output,value)
            print(json.dumps(value,ensure_ascii=False,indent=2));return 0 if value['ready'] else 2
        if args.command == "collect":
            if args.kind == "compile_alignment":
                if args.log_file or args.scheme_file:
                    raise ValueError("--log-file/--scheme-file apply only to model_accuracy")
                value = collect_compile(args.run_dir, args.version_name, expected_frames=args.expected_frames, gt_dir=args.gt_dir)
            else:
                if args.gt_dir:
                    raise ValueError('--gt-dir applies only to compile_alignment')
                value = collect_model(args.run_dir, args.version_name, expected_frames=args.expected_frames,
                                      log_file=args.log_file, scheme_file=args.scheme_file)
            save_json(args.output, value, overwrite=args.overwrite)
            print(json.dumps(summary(value), ensure_ascii=False, indent=2))
            return 0 if value["collection"]["status"] == "complete" else 3
        if args.command == "collect-lpai":
            value = collect_task(args.task_uid, args.output_dir, version_name=args.version_name,
                                 pod_name=args.pod, limit=args.limit)
            print(json.dumps(summary(value), ensure_ascii=False, indent=2))
            return 0 if value["collection"]["status"] == "complete" else 3
        if args.command == "collect-ssh":
            value = collect_remote(args.ssh_alias, args.remote_run_dir, args.kind, args.output_dir,
                                   version_name=args.version_name, python=args.python, max_file_mb=args.max_file_mb)
            print(json.dumps(summary(value), ensure_ascii=False, indent=2))
            return 0 if value["collection"]["status"] == "complete" else 3
        if args.command == "plan":
            value = build_plan(load_json(args.request), args.run_id)
            save_json(args.output, value)
            print(json.dumps({k: value[k] for k in ("run_id", "kind", "ready", "missing_fields", "issues", "steps")},
                             ensure_ascii=False, indent=2))
            return 0 if value["ready"] else 2
        if args.command == "execute":
            value = execute(load_json(args.plan), args.evidence_dir, resume=args.resume, batch_limit=args.batch_limit)
            print(json.dumps(value, ensure_ascii=False, indent=2))
            if value["state"] == "checkpointed":
                return 75
            if value["state"] != "succeeded":
                return 1
            if value.get("collection_error"):
                return 3
            if value.get("result_file") and load_json(value["result_file"])["collection"]["status"] != "complete":
                return 3
            return 0
        if args.command == "prepare-bundle":
            print(json.dumps(prepare_bundle(args.zip, args.destination, args.expected_sha256), ensure_ascii=False, indent=2))
            return 0
        if args.command == 'report-compile' or args.command == 'report' and args.kind == 'compile_alignment' and not args.template_only:
            if not args.result:
                raise ValueError('A single executed compile result is required')
            if args.command=='report' and args.template_file:
                raise ValueError('Compile reports use evaluation semantics, not a release template')
            value = write_compile_report(args.output,result_file=args.result,request_file=args.request_file,
                                         source_dir=args.source_dir,supplemental_dir=args.supplemental_dir,layout=getattr(args,'layout','business'))
            print(json.dumps(value,ensure_ascii=False,indent=2))
            return 0
        if args.command == "report":
            if args.kind=='model_accuracy' and not args.template_only:
                if args.template_file:raise ValueError('Model single reports do not use a release template')
                value=write_model_report(args.output,result_file=args.result)
                print(json.dumps(value,ensure_ascii=False,indent=2))
                return 0
            value = write_report(args.kind, args.output, result_file=args.result, template_only=args.template_only,
                                 readiness_file=args.readiness_file, request_file=args.request_file,
                                 template_file=args.template_file)
            print(json.dumps(value, ensure_ascii=False, indent=2))
            return 0
        if args.command == 'report-model':
            value=write_model_report(args.output,result_file=args.result,binding_file=args.binding_file,
                                     gt_preparation_file=args.gt_preparation_file,head_result_files=args.head_result,
                                     trajectory_figure_file=args.trajectory_figure_receipt,supporting_artifacts=args.supporting_artifact,layout=args.layout)
            print(json.dumps(value,ensure_ascii=False,indent=2))
            return 0
        if args.command == "inspect-datalink":
            value = inspect_datalink(args.file, args.split)
            save_json(args.output, value)
            print(json.dumps({k: value[k] for k in ("source", "sha256", "splits", "scene_count", "frame_count", "issues")},
                             ensure_ascii=False, indent=2))
            return 0 if not value["issues"] else 3
    except (ValueError, OSError, subprocess.SubprocessError, KeyError, TypeError) as error:
        print(json.dumps({"error": str(error), "type": type(error).__name__}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
