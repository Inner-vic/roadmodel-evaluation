"""Read-only reproduction of CLI wrapper failure; never sends a document write."""
import json
from pathlib import Path
import shutil
import subprocess
root=Path(__file__).resolve().parents[1]
cli=shutil.which('lark-cli')
proc=subprocess.run([cli,'--version'],capture_output=True,timeout=25)
value={'command':cli,'exit_code':proc.returncode,'stdout':proc.stdout.decode(errors='replace'),
       'stderr':proc.stderr.decode(errors='replace'),'operation':'version_only'}
report=root/'evidence/feishu-entry-20261029/rm-small-20261029-01/model_accuracy/report/report.xml'
probe=subprocess.run([cli,'docs','+create','--content','@./report.xml','--doc-format','xml','--as','user','--dry-run'],
                     cwd=report.parent,capture_output=True,timeout=30)
value['dry_run']={'exit_code':probe.returncode,'stdout':probe.stdout.decode(errors='replace'),
                  'stderr':probe.stderr.decode(errors='replace')}
out=root/'evidence/feishu-entry-20261029/publication-wrapper-inspection.json'
out.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'version_exit':proc.returncode,'dry_run_exit':probe.returncode,
    'dry_run_stdout_start':value['dry_run']['stdout'][:700],'dry_run_stderr_start':value['dry_run']['stderr'][:700]},ensure_ascii=False))
