import copy
import unittest
from pcm_eval.dataset_preparation import advance
import test_dataset_mount as fixtures


class PreparationTest(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.DatasetMountTest();fixture.setUp()
        self.asset,self.space=fixture.asset,fixture.space
        self.saved=[];self.calls=[]
        def stop(space):self.calls.append('stop');self.space['status']['phase']='stopped'
        def configure(space,asset):
            self.calls.append('configure');self.space['spec']['dataVols'].append(fixture.volume());self.space['status']['phase']='ready'
        def start(space):self.calls.append('start');self.space['status']['phase']='ready'
        self.actions={'assert_idle':lambda s:True,'stop':stop,'configure_and_start':configure,'start':start,
            'probe':lambda p:{'state':'MOUNT_VERIFIED','asset_binding_sha256':self.asset['binding_sha256'],'space_uid':self.space['metadata']['uid'],
                'execution_location':'lpai','workstation_data_used':False,'mount_path':p['mount']['mountPath']}}
    def tick(self,journal):return advance(self.asset,copy.deepcopy(self.space),journal,save=lambda d:self.saved.append(copy.deepcopy(d)),actions=self.actions)
    def test_agent_stops_mounts_starts_and_verifies_without_requester_steps(self):
        j=self.tick({});self.assertEqual(self.calls,['stop'])
        j=self.tick(j);self.assertEqual(self.calls,['stop','configure'])
        j=self.tick(j);self.assertEqual(j['state'],'DATA_READY')
        j=self.tick(j);self.assertEqual(self.calls,['stop','configure'])
    def test_busy_space_is_waited_for_not_stopped(self):
        self.actions['assert_idle']=lambda s:False
        self.assertEqual(self.tick({})['state'],'PREPARING');self.assertEqual(self.calls,[])
    def test_mutation_intent_is_saved_before_action_and_unknown_result_not_retried(self):
        def unknown(s):
            self.assertEqual(self.saved[-1]['intent']['action'],'stop');self.calls.append('unknown');raise TimeoutError()
        self.actions['stop']=unknown
        j=self.tick({});self.assertEqual(j['state'],'NEEDS_REVIEW')
        self.tick(j);self.assertEqual(self.calls,['unknown'])
    def test_unrelated_configuration_change_blocks_evaluation(self):
        self.space['status']['phase']='stopped';j=self.tick({});self.space['spec']['image']='different'
        self.assertEqual(self.tick(j)['state'],'NEEDS_REVIEW')
    def test_missing_runtime_action_is_not_a_user_operation_request(self):
        del self.actions['stop'];j=self.tick({});self.assertEqual(j['state'],'NEEDS_REVIEW');self.assertNotIn('intent',j)
    def test_create_uses_a_bound_template_and_is_never_duplicated(self):
        actions={**self.actions,'environment_template_sha256':'template-hash','create':lambda a:self.calls.append('create')}
        tick=lambda j:advance(self.asset,None,j,save=lambda d:self.saved.append(copy.deepcopy(d)),actions=actions)
        j=tick({});self.assertEqual(self.calls,['create']);self.assertEqual(j['intent']['action'],'create')
        self.assertEqual(tick(j)['state'],'NEEDS_REVIEW');self.assertEqual(self.calls,['create'])


if __name__=='__main__':unittest.main()
