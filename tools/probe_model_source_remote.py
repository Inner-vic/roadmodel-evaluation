import pathlib,subprocess,json
root=pathlib.Path('/lpai/bmad-roadmodel-source/8a29aad33330fafc98aa248a17ade520452eea69')
out={'root':str(root),'files': [p.name for p in root.iterdir()]}
out['git']=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
out['ops']=[str(p) for p in root.rglob('*.so')][:100]
out['mounts']={str(p):[c.name for c in p.iterdir()][:30] for p in [pathlib.Path('/lpai/models'),pathlib.Path('/lpai/dataset'),pathlib.Path('/lpai/inputs')] if p.exists()}
print(json.dumps(out))
