"""Parse recorded launch settings without executing any shell content."""
import re
import shlex

FLAGS={'py-config','config','resume-from','load-from','traj-resume','traj-load',
       'classifier-resume','classifier-load','skip-module-names','work-dir'}
SECRET=re.compile(r'TOKEN|PASSWORD|SECRET|API_KEY|CREDENTIAL',re.I)

def parse_launch(command):
    # shlex excludes comments; continuations join only recorded source text.
    logical=command.replace('\\\n',' ')
    env={}; flags={}
    for line in logical.splitlines():
        tokens=shlex.split(line,comments=True,posix=True)
        if not tokens:continue
        if tokens[0]=='export':
            for token in tokens[1:]:
                match=re.fullmatch(r'([A-Z][A-Z0-9_]*)=(.*)',token)
                if match and not SECRET.search(match[1]):env[match[1]]=match[2]
        for i,token in enumerate(tokens):
            if not token.startswith('--'):continue
            key,sep,value=token[2:].partition('=')
            if key not in FLAGS:continue
            if not sep:
                if i+1>=len(tokens) or tokens[i+1].startswith('--'):continue
                value=tokens[i+1]
            flags[key]=value
    return {'environment':env,'flags':flags}
