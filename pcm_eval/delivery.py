"""Portable downstream report delivery; never runs evaluation or publication."""
from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath

from .common import digest, load_json, save_json, utc_now


def export_delivery(destination, *, kind, report_json, report_xml, evidence_bundle, results=(), sections=None):
    dest = Path(destination).resolve()
    if dest.exists():
        raise ValueError('Delivery destination exists; preserve prior deliveries')
    report = load_json(report_json)
    if report.get('evaluation_type') != kind:
        raise ValueError('Delivery evaluation type differs from report')
    entries = [('report_json', Path(report_json), 'report/report.json'),
               ('report_xml', Path(report_xml), 'report/report.xml'),
               ('evidence_bundle', Path(evidence_bundle), 'evidence/' + Path(evidence_bundle).name)]
    for i, source in enumerate(results):
        entries.append(('source_result', Path(source), f'results/{i:02d}.json'))
    if sections:
        index = load_json(sections)
        if index.get('report_sha256') != digest(Path(report_json)):
            raise ValueError('Section index describes another report')
        for i, section in enumerate(index['sections']):
            source = Path(section['path'])
            if digest(source) != section['sha256']:
                raise ValueError('Report section changed')
            entries.append(('section', source, f'report/sections/{i:02d}-{section["id"]}.xml'))
    if any(not source.is_file() for _, source, _ in entries):
        raise ValueError('A delivery source file is missing')
    dest.mkdir(parents=True)
    files = []
    for role, source, relative in entries:
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        expected = digest(source)
        if digest(target) != expected:
            raise ValueError('Source changed during export')
        files.append({'role': role, 'path': relative, 'bytes': target.stat().st_size, 'sha256': expected})
    manifest = {'schema': 'roadmodel.delivery/v1', 'evaluation_type': kind, 'created_at': utc_now(),
                'files': files, 'entrypoints': {'report': 'report/report.json', 'xml': 'report/report.xml'},
                'states': {'report': 'generated', 'publication': 'not_requested', 'quality': 'not_assessed'},
                'consumption': {'report_json': 'self_contained_metric_values_and_provenance',
                                'source_result_paths': 'historical_provenance; not execution paths on the receiving host',
                                'publication': 'render again using current package paths before publishing',
                                'rerun_evaluation': False}}
    save_json(dest / 'manifest.json', manifest)
    return inspect_delivery(dest)


def inspect_delivery(directory):
    root = Path(directory).resolve()
    manifest = load_json(root / 'manifest.json')
    if manifest.get('schema') != 'roadmodel.delivery/v1':
        raise ValueError('Unknown delivery schema')
    seen = set()
    for item in manifest['files']:
        relative = PurePosixPath(item['path'])
        if relative.is_absolute() or '..' in relative.parts or '\\' in item['path'] or ':' in item['path'] or item['path'] in seen:
            raise ValueError('Unsafe or duplicate delivery path')
        seen.add(item['path'])
        target = (root / item['path']).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError('Delivery file missing or outside package')
        if target.stat().st_size != item['bytes'] or digest(target) != item['sha256']:
            raise ValueError('Delivery content hash mismatch')
    entry = manifest['entrypoints']['report']
    if entry not in seen:
        raise ValueError('Unlisted report entrypoint')
    report = load_json(root / entry)
    if report.get('evaluation_type') != manifest['evaluation_type']:
        raise ValueError('Manifest and report evaluation types differ')
    return {'state': 'verified', 'manifest': str(root / 'manifest.json'),
            'evaluation_type': manifest['evaluation_type'], 'files': len(seen),
            'report': str(root / entry), 'report_schema': report['schema'], 'new_inference_frames': 0}
