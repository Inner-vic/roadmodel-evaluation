import copy
import json
import unittest
from tools.notebook_sdk_worker import safe_error,validate_body
from tests.test_installation import request

class NotebookSdkTest(unittest.TestCase):
    def setUp(self):
        self.request=request();self.request['dataset_ref']='https://lpai.lixiang.com/lpaiweb/xspacex_team/subApp/dataset/detail/dataset-id?version=v1'
        t=self.request['template'];key=self.request['ssh']['credential_name'];path='/mnt/datasets/test-data/v1'
        self.body={'metadata':{'name':'dedicated','namespace':'team','labels':{'lpai.org/kind':'notebook','lpai.org/visible':'personal'},
            'annotations':{'lpai.org/desc':'Dedicated evaluation','lpai.org/gems-project':t['project']}},
            'spec':{'image':t['image'],'clusterName':t['cluster'],'resources':t['resources'],'sshSecretName':key,
                'autoCulling':{'enable':'true','windowMinutes':'240'},'env':{},
                'dataVols':[{'type':'dataset','id':'dataset-id','name':'test-data','version':'v1','fsType':'ofs',
                    'mountPath':path,'envKey':'LPAI_INPUT_DATASET_0','envValue':path}],
                'accesses':[{'name':'jupyter-lab'},{'name':'ssh','pubkeyName':key}]}}

    def test_controlled_gpu_template_accepts_exact_input_mount(self):
        validate_body(self.request,self.body)

    def test_wrong_dataset_or_environment_injection_rejected(self):
        for field,value in (('id','another-dataset'),('version','latest'),('mountPath','/tmp/other')):
            changed=copy.deepcopy(self.body);changed['spec']['dataVols'][0][field]=value
            with self.assertRaises(AssertionError):validate_body(self.request,changed)
        self.body['spec']['env']={'TOKEN':'not-allowed'}
        with self.assertRaises(AssertionError):validate_body(self.request,self.body)

    def test_error_response_retains_cause_without_echoed_credentials(self):
        exc=ValueError('Invalid status code \'500\' from route with response: '+json.dumps({
            'code':'InvalidName','message':'name is too long','authorization':'secret','request':{'token':'secret'}}))
        value=safe_error(exc)
        self.assertEqual(value['http_status'],500)
        self.assertEqual(value['response'],{'code':'InvalidName','message':'name is too long'})
        self.assertNotIn('secret',json.dumps(value))
        value=safe_error(ValueError('Invalid status code \'500\' from route with response: '+json.dumps({'message':'LpaiJwt secret'})))
        self.assertNotIn('secret',json.dumps(value))

if __name__=='__main__':unittest.main()
