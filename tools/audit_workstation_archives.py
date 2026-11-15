"""Inspect archive indexes; never unpack entire archives or change source files."""
import collections
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

OUT = Path('/mnt/data/developer/pcm-roadmodel-eval/dataset-audit')


def main():
    discovery = json.loads((OUT / 'discovery.json').read_text())
    inventory = json.loads((OUT / 'expected-inventory.json').read_text())
    expected = collections.defaultdict(list)
    scenes = set()
    for item in inventory['files']:
        expected[(Path(item['path']).name, item['size'])].append(item['path'])
        if '/dataset/' in item['path']:
            scenes.add(item['path'].split('/')[0])
    results = []
    for candidate in discovery['archive_candidates']:
        path = Path(candidate['path'])
        lower = path.name.lower()
        result = {'path': str(path), 'bytes': candidate['size'], 'expected_scene_matches': [],
                  'name_size_candidates': [], 'members_scanned': 0, 'sample': []}
        matches = set()

        def add(name, size):
            result['members_scanned'] += 1
            parts = name.replace('\\', '/').split('/')
            matches.update(set(parts).intersection(scenes))
            rows = expected.get((parts[-1], size), [])
            if rows:
                result['name_size_candidates'].append({'member': name, 'size': size,
                                                       'expected_paths': rows})
            if len(result['sample']) < 3:
                result['sample'].append(name)

        try:
            if lower.endswith('.7z'):
                proc = subprocess.run(['7z', 'l', '-slt', '-ba', '--', str(path)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
                if proc.returncode:
                    result['state'] = 'index_unreadable'
                else:
                    for block in proc.stdout.decode(errors='replace').split('\n\n'):
                        fields = dict(line.split(' = ', 1) for line in block.splitlines() if ' = ' in line)
                        if fields.get('Path'):
                            add(fields['Path'], int(fields.get('Size', '0')))
                    result['state'] = 'index_complete'
            elif lower.endswith('.zip'):
                with zipfile.ZipFile(path) as archive:
                    for member in archive.infolist():
                        add(member.filename, member.file_size)
                result['state'] = 'index_complete'
            elif lower.endswith(('.tar.gz', '.tgz')):
                # Full listing of raw_pack_11 is relevant to the July V7 scenes.
                # Other compressed recordings get a bounded prefix check only.
                full = lower == 'raw_pack_11.tar.gz'
                with tarfile.open(str(path), 'r|gz') as archive:
                    for member in archive:
                        add(member.name, member.size)
                        if not full and result['members_scanned'] >= 6:
                            break
                result['state'] = 'index_complete' if full else 'prefix_sample_only'
            elif lower.endswith('.tar'):
                with tarfile.open(str(path), 'r:') as archive:
                    for member in archive:
                        add(member.name, member.size)
                result['state'] = 'index_complete'
            else:
                result['state'] = 'unsupported_index'
        except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile, subprocess.TimeoutExpired) as error:
            result.update(state='index_failed', error_type=type(error).__name__)
        result['expected_scene_matches'] = sorted(matches)
        results.append(result)
        (OUT / 'archive-audit.json').write_text(json.dumps({'archives': results}, indent=2))
    summary = {'archives_examined': len(results),
               'states': dict(collections.Counter(x['state'] for x in results)),
               'matches': [{'path': r['path'], 'expected_scene_matches': r['expected_scene_matches'],
                            'name_size_candidate_count': len(r['name_size_candidates'])}
                           for r in results if r['expected_scene_matches'] or r['name_size_candidates']]}
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
