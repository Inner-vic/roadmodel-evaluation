import copy
import json
from pathlib import Path
import tempfile
import unittest

from pcm_eval.dataset_mount import dataset_reference, resolve_dataset, resolve_dataset_input, mount_plan, probe_mounted_index


URL = 'https://lpai.lixiang.com/lpaiweb/xspacex_team/subApp/dataset/detail/' + 'a'*32 + '?version=new-42'


class DatasetMountTest(unittest.TestCase):
    def setUp(self):
        self.raw = {'metadata': {'uid': 'version-new', 'pid': 'a'*32, 'pName': 'roads', 'name': 'new-42',
                      'ownerReferences': [{'kind': 'namespace', 'uid': 'team'}],
                      'labels': {'status': 'AVAILABLE'}}, 'spec': {'dir': {'path': 's3://bucket/new/'}}}
        self.asset = resolve_dataset(URL, lambda a: self.raw)
        self.space = {'metadata': {'uid': 'space-1', 'name': 'mine', 'namespace': 'team'},
                      'spec': {'dataVols': []}, 'status': {'phase': 'ready'}}
    def volume(self, root='/mnt/roads'):
        return {'type': 'dataset', 'name': 'roads', 'version': 'new-42', 'mountPath': root,
                'mode': 'ReadOnly', 'pvcName': 'actual-platform-pvc'}
    def test_supplied_version_is_queried_without_catalog_or_default(self):
        calls=[]
        out=resolve_dataset(URL, lambda args: calls.append(args) or self.raw)
        self.assertEqual(calls[0][-2:], ['--version','new-42'])
        self.assertEqual(out['source_uri'], 's3://bucket/new/')
    def test_no_implicit_latest_or_ambiguous_version(self):
        for suffix in ['', '?version=', '?version=new&version=old']:
            with self.assertRaises(ValueError):dataset_reference(URL.split('?')[0]+suffix)
    def test_space_path_resolves_only_its_explicit_platform_volume(self):
        self.space['spec']['dataVols']=[self.volume()]
        a=resolve_dataset_input('开发空间:/mnt/roads/data_link_current.json',lambda a:self.raw,self.space)
        self.assertEqual(a['source_ref'],'开发空间:/mnt/roads/data_link_current.json')
        self.assertEqual(a['version'],'new-42');self.assertEqual(a['index_relative_hint'],'data_link_current.json')
    def test_workstation_unknown_and_scene_paths_do_not_select_a_default(self):
        self.space['spec']['dataVols']=[self.volume()]
        for value in ['工作站:/mnt/roads','/unknown','/mnt/roads/scene']:
            with self.assertRaises(ValueError):resolve_dataset_input(value,lambda a:self.raw,self.space)
    def test_different_asset_uid_or_namespace_is_rejected(self):
        for key, value in [('pid','b'*32), ('name','old')]:
            bad=copy.deepcopy(self.raw);bad['metadata'][key]=value
            with self.assertRaises(ValueError):resolve_dataset(URL, lambda a: bad)
        self.space['metadata']['namespace']='other'
        with self.assertRaises(ValueError):mount_plan(self.asset,self.space)
    def test_old_space_mount_never_substitutes_requested_asset(self):
        old=self.volume();old['version']='old';self.space['spec']['dataVols']=[old]
        before=copy.deepcopy(self.space)
        plan=mount_plan(self.asset,self.space)
        self.assertEqual(plan['state'],'SPACE_STOP_REQUIRED')
        self.assertFalse(plan['evaluation_allowed']);self.assertEqual(before,self.space)
        self.assertNotIn('mount',plan)
    def test_stopped_space_needs_actual_platform_mount_not_preheat_guess(self):
        self.space['status']['phase']='stopped'
        plan=mount_plan(self.asset,self.space)
        self.assertEqual(plan['state'],'MOUNT_REQUIRED');self.assertNotIn('pvcName',str(plan))
    def test_configured_mount_still_requires_probe(self):
        self.space['spec']['dataVols']=[self.volume()]
        plan=mount_plan(self.asset,self.space)
        self.assertEqual(plan['state'],'MOUNT_CONFIGURED');self.assertFalse(plan['evaluation_allowed'])
        self.space['spec']['dataVols'][0]['mode']='Writable'
        with self.assertRaises(ValueError):mount_plan(self.asset,self.space)
    def test_actual_readonly_mount_and_dynamic_index_are_checked(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'mounted';root.mkdir()
            index=root/'data_link_new_42.json';index.write_text(json.dumps({'eval':{'s':{'frames':[{'frame_index':7}]}},'vision_eval':{}}))
            self.space['spec']['dataVols']=[self.volume(root.as_posix())]
            plan=mount_plan(self.asset,self.space)
            info=Path(td)/'mountinfo';info.write_text('1 2 0:1 / '+root.as_posix()+' ro - fuse.ofs actual ro\n')
            proof=probe_mounted_index(plan,mountinfo_path=info)
            self.assertEqual(proof['index_relative'],'data_link_new_42.json')
            self.assertEqual(proof['split_counts']['eval']['frames'],1)
            self.assertFalse(proof['workstation_data_used']);self.assertFalse(proof['evaluation_allowed'])
            (root/'data_link_other.json').write_text('{}')
            with self.assertRaises(ValueError):probe_mounted_index(plan,mountinfo_path=info)
            (root/'data_link_other.json').unlink()
            info.write_text('1 2 0:1 / '+root.as_posix()+' rw - ext4 actual rw\n')
            with self.assertRaises(ValueError):probe_mounted_index(plan,mountinfo_path=info)
    def test_local_cache_is_not_mount_and_index_cannot_escape(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'cache';root.mkdir()
            self.space['spec']['dataVols']=[self.volume(root.as_posix())]
            plan=mount_plan(self.asset,self.space);info=Path(td)/'mountinfo';info.write_text('')
            with self.assertRaises(ValueError):probe_mounted_index(plan,mountinfo_path=info)


if __name__=='__main__':unittest.main()
