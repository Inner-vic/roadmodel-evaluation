import pathlib,subprocess,json,os
root=pathlib.Path('/lpai/bmad-roadmodel-source/8a29aad33330fafc98aa248a17ade520452eea69')
out={'root':str(root),'files':[p.name for p in root.iterdir()]}
out['git']=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
out['ops']=[]
for d,dirs,files in os.walk(root):
 dirs[:]=[x for x in dirs if x not in ['.git','node_modules','__pycache__','data','outputs','vis','checkpoint','evidence','inspection','runs','out','logs']]
 if len(pathlib.Path(d).relative_to(root).parts)>7:dirs[:]=[];continue
 out['ops'] += [str(pathlib.Path(d)/f) for f in files if f.endswith('.so')]
out['mounts']={str(p):[c.name for c in p.iterdir()][:30] for p in [pathlib.Path('/lpai/models'),pathlib.Path('/lpai/dataset'),pathlib.Path('/lpai/inputs')] if p.exists()}
print(json.dumps(out))
