"""Run a confirmed, registry-bound model ticket using the verified native code."""
from pathlib import Path
import copy
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from pcm_eval.common import digest


def main(run):
    run=Path(run).resolve();request=json.loads((run/'request.json').read_text());resolved=request['resolved'];cfg=request['runtime_config']
    assert run.name==request['ticket_id'] and resolved['ready'] and not resolved['compile_enabled']
    assert cfg.get('execution_location')!='lpai','旧执行器包含工作站数据准备，不能用于LPAI全流程工单'
    assert resolved.get('single_dataset') and resolved.get('scope')=='full'
    assert len({h['dataset_id'] for h in resolved['heads'].values()})==1
    assert all(cfg['native_dataset_bindings'].get(k)==h['dataset_id'] for k,h in resolved['heads'].items()),'单数据入口尚未绑定全部原生评测项，不启动推理'
    assert digest(run/'request.json')==json.loads((run/'dispatch.json').read_text())['request_sha256']
    def progress(stage,state='RUNNING',**extra):
        p=run/'progress.json';q=run/'progress.tmp';q.write_text(json.dumps({'state':state,'stage':stage,'resolved':resolved,**extra},ensure_ascii=False));q.replace(p)
    try:
        progress('正在核验模型与数据绑定')
        original=Path(cfg['native_release']);private=run/'native-code';private.mkdir()
        shutil.copytree(original/'tools',private/'tools');shutil.copytree(original/'pcm_eval',private/'pcm_eval')
        shutil.copytree(original/'configs',private/'configs')
        # Only two small launch descriptors are needed; never copy datasets,
        # checkpoints, old predictions or the whole historical evidence tree.
        for name in ['evidence/model-evaluation/model-reference-20261027/native-launch.json',
                     'evidence/model-evaluation/remaining-heads-20261027/vision-launch.json']:
            p=private/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original/name,p)
        # Controllers are the previously validated code. Only task identity,
        # frozen frame count, isolated output directory and data-link path vary.
        n=len(resolved['heads']['occ']['frames']);v=len(resolved['heads'].get('vision',{}).get('frames',[]))
        space=cfg['space_parent']+'/tickets/'+run.name;oldspace=cfg['old_space']
        for name in ['formal_model_controller.py','formal_vision_controller.py']:
            p=private/'tools'/name;s=p.read_text()
            s=s.replace("SPACE = '"+cfg['timed_space']+"'",'SPACE = '+repr(space))
            assert 'SPACE = '+repr(space) in s
            s=s.replace('model-timing-v322-20261102-r1',run.name)
            if name=='formal_model_controller.py':s=s.replace('8337',str(n))
            else:
                s=s.replace('11543',str(v))
                s=s.replace(repr(oldspace+'/vision-input/full-data-link.json'),repr(space+'/vision-input/data-link.json'))
            compile(s,str(p),'exec');p.write_text(s)
        (run/'model').mkdir();(run/'vision').mkdir()
        frames=resolved['heads']['occ']['frames'];vision_frames=resolved['heads'].get('vision',{}).get('frames',[])
        frozen={'task_id':run.name,'model_version':resolved['model']['version'],'frames':frames,'expected_frames':n,
                'data_link_sha256':resolved['heads']['occ']['index_sha256'],'vision':{'frames':vision_frames,'expected_frames':v}}
        (run/'native-request.json').write_text(json.dumps(frozen))
        # Native controllers read request.json in their task root. Keep the user
        # work-order request immutable in its parent and create a separate task.
        task=run/'evaluation';task.mkdir();(task/'model').mkdir();(task/'vision').mkdir()
        (task/'request.json').write_text(json.dumps(frozen))
        (task/'model/gpu-selection.json').write_text(json.dumps(cfg['gpu_selection']))
        (task/'vision-data-state.json').write_text(json.dumps({'state':'READY','cache_receipt':cfg['vision_cache_receipt']}))
        os.environ.update(ROADMODEL_TASK_ROOT=str(task),ROADMODEL_STAGE_LOCAL='1',ROADMODEL_IDENTITY=cfg['identity_file'])
        sys.path[:0]=[str(private),str(private/'tools')]
        # Import native modules in this process only; the shared Bot and old
        # completed runs are never patched or restarted by a ticket.
        for module in ['formal_model_controller','formal_vision_controller','model_space_transport','formal_gpu']:
            sys.modules.pop(module,None)
        occ=importlib.import_module('formal_model_controller')
        assert occ.SPACE==space and occ.TASK.resolve()==task
        progress('正在检查开发空间与GPU可用性')
        gpu=cfg['gpu_selection']['physical_index']
        probe=json.loads(occ.run(['python3','-c',occ.PROBE,str(gpu)],timeout=45).decode())
        if probe['free_GiB']<6:raise ValueError('已登记GPU当前显存不足；未开始评测，请核验后再继续')
        # Native workers import the same timing support as the validated run.
        occ.put(private/'tools/model_rerun_timing.py',space+'/model_rerun_timing.py')
        if v:
            occ.put(task/'request.json',space+'/request.json')
            make="""import pathlib,json,sys
src=pathlib.Path(sys.argv[1]);dst=pathlib.Path(sys.argv[2]);req=json.loads(pathlib.Path(sys.argv[3]).read_text());data=json.loads(src.read_text());selected=set(map(tuple,req['vision']['frames']));out={k:{} for k in data};actual=[]
for s,info in data['vision_eval'].items():
 fs=[f for f in info['frames'] if (s,f['frame_index']) in selected]
 if fs:out['vision_eval'][s]=dict(info,frames=fs);actual.extend([[s,f['frame_index']] for f in fs])
assert actual==req['vision']['frames'],'Visual index scope/order differs'
dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();dst.write_text(json.dumps(out));print(len(actual))
"""
            assert int(occ.run(['python3','-c',make,oldspace+'/vision-input/full-data-link.json',space+'/vision-input/data-link.json',space+'/request.json']).decode())==v
        progress('正在执行OCC模型精度评测')
        occ.main()
        if v:
            progress('OCC已完成，正在执行视觉分类模型精度评测')
            vision=importlib.import_module('formal_vision_controller');assert vision.SPACE==space;vision.main()
        progress('正在按原PCM模板整理模型结果')
        from pcm_eval.model_aggregate import aggregate_occ
        from pcm_eval.model_report import build_model_report
        from pcm_eval.strict_model_template import model_tables,render
        from pcm_eval.reference_report import display
        files=[task/'model'/f'batch-{i:04d}-a02/normalized-result.json' for i in range((n+99)//100)]
        result=aggregate_occ(files,frames,run/'report/occ',run_id=run.name+'-occ')
        baseline=json.loads(Path(cfg['baseline_model_report']).read_text());binding=copy.deepcopy(baseline['binding']);binding['dataset'].update(selected_frames=frames,version=resolved['heads']['occ']['version'])
        extra=[json.loads((task/'vision/normalized-result.json').read_text())] if v else []
        model=build_model_report(result,binding=binding,head_results=extra)
        (run/'report/model-report.json').write_text(json.dumps(model,ensure_ascii=False,indent=2))
        ref=json.loads(Path(cfg['reference_file']).read_text())['data']['document']['content']
        spec=json.loads((ROOT/'configs/report_template.json').read_text())['model_accuracy']
        tables=model_tables(model,ref,spec);body=render(ref,tables,model_version=model['version_name'],vision_samples=v)
        publication=run/'publication';publication.mkdir();(publication/'body.xml').write_text(body,encoding='utf8')
        expected=[[t['headers']]+[[display(c) for c in row] for row in t['rows']] for t in tables]
        def vals(root):return [[[''.join(c.itertext()) for c in row] for row in t.iter('tr')] for t in root.iter('table')]
        seq=0
        def cli(args):
            nonlocal seq
            seq+=1;p=subprocess.run([cfg['lark_cli'],*args,'--as','bot'],cwd=publication,
                env={**os.environ,'LARKSUITE_CLI_CONFIG_DIR':cfg['lark_config'],'LARKSUITE_CLI_NO_UPDATE_NOTIFIER':'1','LARKSUITE_CLI_NO_SKILLS_NOTIFIER':'1'},capture_output=True,timeout=180)
            (publication/f'{seq}.stdout.json').write_bytes(p.stdout);(publication/f'{seq}.stderr.txt').write_bytes(p.stderr)
            d=json.loads(p.stdout or p.stderr);assert p.returncode==0 and d.get('ok'),'飞书操作需核验原收据，不自动重试'
            d=d['data'];assert not d.get('warnings') and d.get('result') not in ('failed','partial_success');return d
        doc=request['delivery']['document_id'];before=cli(['docs','+fetch','--doc',doc,'--detail','full'])['document'];old=ET.fromstring('<root>'+before['content']+'</root>')
        (publication/'before.json').write_text(json.dumps(before,ensure_ascii=False))
        oldtables=vals(old);assert len(oldtables)==3 and all(a[0]==b[0] and [r[:2] if a[0][0]=='父类' else r[:1] for r in a[1:]]==[r[:2] if b[0][0]=='父类' else r[:1] for r in b[1:]] for a,b in zip(oldtables,expected)),'报告模板已变更，停止写入'
        nodes=[n for n in old if n.tag!='title'];(publication/'journal.json').write_text(json.dumps({'state':'WRITING','revision':before['revision_id']}))
        cli(['docs','+update','--doc',doc,'--command','block_replace','--start-block-id',nodes[0].get('id'),'--end-block-id',nodes[-1].get('id'),'--content','@./body.xml','--doc-format','xml','--revision-id',str(before['revision_id'])])
        after=cli(['docs','+fetch','--doc',doc,'--detail','full'])['document'];assert vals(ET.fromstring('<root>'+after['content']+'</root>'))==expected
        receipt={'state':'VERIFIED','report_url':'https://li.feishu.cn/docx/'+doc,'revision_id':after['revision_id'],'ticket_id':run.name,'model_table_count':3,'all_model_table_cells_match':True}
        (publication/'verification.json').write_text(json.dumps(receipt));progress('模型精度评测及报告回读完成','SUCCEEDED',report_url=receipt['report_url'])
    except Exception as error:
        progress('需要核验本次执行收据','NEEDS_REVIEW',issues=[str(error)[:300]])
        raise


if __name__=='__main__':main(sys.argv[1])
