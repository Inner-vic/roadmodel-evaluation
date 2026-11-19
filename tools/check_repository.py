"""Check checkout syntax and local documentation without importing application code."""
import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]

def main():
    ignored = {'.git', '__pycache__', '.pytest_cache', '.venv', '.evidence'}
    paths = [p for p in ROOT.rglob('*') if p.is_file()
             and not ignored.intersection(p.relative_to(ROOT).parts)]
    errors = []
    counts = {'python': 0, 'json': 0, 'links': 0}
    for path in paths:
        name = path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            errors.append(name + ': linked source is not accepted')
            continue
        raw = path.read_bytes()
        if re.search(rb'(?m)^-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', raw):
            errors.append(name + ': private key header')
        if re.search(rb'\b(?:glpat-|gh[pousr]_|github_pat_|sk-proj-)[A-Za-z0-9_-]{20,}', raw):
            errors.append(name + ': possible access token')
        try:
            if path.suffix in {'.py', '.pyw'}:
                ast.parse(raw.decode('utf-8-sig'), filename=name)
                counts['python'] += 1
            elif path.suffix == '.json':
                json.loads(raw.decode('utf-8-sig'))
                counts['json'] += 1
            elif path.suffix == '.md':
                text = re.sub(r'```[^\n]*\n.*?```', '', raw.decode('utf-8-sig'), flags=re.S)
                for target in re.findall(r'!?\[[^\]\n]*\]\(([^\n]*?)\)', text):
                    target = target.strip().split(' "', 1)[0].strip('<>')
                    url = urlsplit(target)
                    if not target or target.startswith('#') or url.scheme or url.netloc:
                        continue
                    destination = (path.parent / unquote(url.path)).resolve()
                    counts['links'] += 1
                    if not destination.is_relative_to(ROOT) or not destination.exists():
                        errors.append(name + ': missing local link: ' + target)
        except (SyntaxError, ValueError, UnicodeError) as error:
            errors.append(name + ': ' + type(error).__name__)
    print(json.dumps({'status': 'failed' if errors else 'passed', **counts,
                      'errors': errors, 'application_imports': 0}))
    return bool(errors)

if __name__ == '__main__':
    raise SystemExit(main())
