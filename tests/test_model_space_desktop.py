import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import model_space_desktop as desktop
import test_dataset_mount as fixture_module


class DesktopPreparationTest(unittest.TestCase):
    def test_mounted_dataset_uses_real_probe_without_starting_browser(self):
        f=fixture_module.DatasetMountTest();f.setUp();f.space['spec']['dataVols']=[f.volume()];f.space['spec']['clusterName']='cluster'
        profile={'namespace':'team','space_name':'mine','space_uid':'space-1','cluster':'cluster'}
        job={'id':'job','lease':'lease','request':{'profile':profile,'inputs':{'dataset_ref':fixture_module.URL}}}
        updates=[]
        def query(args,**kwargs):return f.raw if args[:2]==['assets','datasets'] else f.space
        def rpc(cfg,p):
            if p['op']=='update':updates.append(p);return {'ok':True}
            self.assertEqual(p['op'],'probe');plan=p['plan']
            receipt={'state':'MOUNT_VERIFIED','asset_binding_sha256':plan['asset']['binding_sha256'],'space_uid':'space-1',
                     'execution_location':'lpai','workstation_data_used':False,'mount_path':'/mnt/roads','split_counts':{'eval':{'scenes':1,'frames':1}}}
            receipt['dataset_source']={'mount_receipt':copy.deepcopy(receipt),'source_ref':fixture_module.URL}
            return receipt
        with patch.object(desktop,'lpai_request',side_effect=query),patch.object(desktop,'remote',side_effect=rpc),patch.object(desktop,'run_browser_action',side_effect=AssertionError('Already mounted; no UI mutation allowed')):
            result=desktop.process({'lpai_cli':'cli'},'client',job)
        self.assertEqual(result['state'],'DATA_READY');self.assertEqual(updates[-1]['state'],'DATA_READY')
        self.assertEqual(result['new_inference_frames'],0)


if __name__=='__main__':unittest.main()
