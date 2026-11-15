"""Freeze retained small-result bindings and recheck the existing Feishu tables."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcm_eval.common import load_json, save_json, digest, utc_now
from pcm_eval.publication import cli_command
from pcm_eval.reference_report import display

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/trial-entry-20261030/package'
REMOTE='/mnt/data/bench-eval/service/roadmodel/trial-20261030'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sources={
      'retained/model-report.json':'evidence/coverage-supplement-20261029/bound-model.json',
      'retained/compile-report.json':'evidence/coverage-supplement-20261029/compile-report/report.json',
      'retained/combined-data.json':'evidence/coverage-supplement-20261029/combined/data.json',
      'retained/reference.json':'evidence/source-docs/release-template-review-20261029.json',
      'retained/publication.json':'evidence/coverage-supplement-20261029/publication/verification.json',
      'retained/mapping-audit.json':'evidence/coverage-supplement-20261029/mapping-audit.json',
      'retained/vision-audit.json':'evidence/coverage-supplement-20261029/vision-audit.json',
    }
    for n,src in sources.items():
        p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/src,p)
    fetch=OUT/'retained/readback-20261030.json'
    if not fetch.exists():
        proc=subprocess.run([*cli_command(),'docs','+fetch','--doc','DEMO00000000000000000000002','--detail','full','--as','user'],capture_output=True,timeout=180)
        if proc.returncode:raise RuntimeError('Existing report readback failed')
        d=json.loads(proc.stdout.decode('utf-8-sig'));assert d.get('ok');save_json(fetch,d)
    doc=load_json(fetch)['data']['document'];tree=ET.fromstring('<root>'+doc['content']+'</root>')
    actual=[[[ ''.join(c.itertext()) for c in row] for row in t.findall('.//tr')] for t in tree.iter('table')]
    data=load_json(OUT/'retained/combined-data.json')
    expected=[[t['headers'],*[[display(c) for c in row] for row in t['rows']]] for t in data['tables']]
    assert actual==expected and len(actual)==5,'Published five tables changed'
    def rec(n):return {'path':n,'sha256':digest(OUT/n)}
    pub=load_json(OUT/'retained/publication.json');pub.update(revision_id=doc['revision_id'],source_json_sha256=digest(OUT/'retained/combined-data.json'),checked_at=utc_now())
    save_json(OUT/'retained/publication-current.json',pub,overwrite=(OUT/'retained/publication-current.json').exists())
    cat=load_json(ROOT/'configs/two_link_catalog.json');cat['material_root']=REMOTE
    for kind,name in [('model_accuracy','model'),('compile_alignment','compile')]:
        report=load_json(OUT/f'retained/{name}-report.json')
        run=report['run_id'] if name=='model' else report['identity']['run_id']
        scope={'occ_frames':2,'vision_frames':4,'trajectory_frames':2} if name=='model' else {'frames':3,'tensor_pairs':54,'GT_classes':8}
        receipt={'schema':'roadmodel.retained-binding/v1','state':'VERIFIED','kind':kind,'version':'v3.2.2',
                 'dataset_id':cat['datasets'][0]['id'],'new_inference_frames':0,'source_run_id':run,'scope':scope,
                 'atomic_report':rec(f'retained/{name}-report.json'),
                 'evidence':[rec(n) for n in ('retained/mapping-audit.json','retained/vision-audit.json','retained/publication-current.json','retained/readback-20261030.json')]}
        filename=f'retained/{name}-binding.json';save_json(OUT/filename,receipt,overwrite=(OUT/filename).exists())
        cat['models'][0]['chains'][kind]={'version':'v3.2.2','dataset_id':cat['datasets'][0]['id'],
               'heads':['occ','vision','trajectory'] if name=='model' else ['tensor_alignment','gt'],
               'retained_result':rec(filename)}
    delivery={'combined_data':rec('retained/combined-data.json'),'reference':rec('retained/reference.json'),
              'publication':rec('retained/publication-current.json')}
    save_json(OUT/'retained/delivery-binding.json',delivery,overwrite=(OUT/'retained/delivery-binding.json').exists())
    cat['retained_delivery']=rec('retained/delivery-binding.json')
    save_json(OUT/'configs/two_link_catalog.json',cat,overwrite=(OUT/'configs/two_link_catalog.json').exists())
    print(json.dumps({'five_tables_match':True,'revision':doc['revision_id'],'new_inference_frames':0,'report_url':pub['report_url']}))

if __name__=='__main__':main()
