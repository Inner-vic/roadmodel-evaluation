import unittest
from pcm_eval.model_inputs import parse_launch

class LaunchInputTests(unittest.TestCase):
 def test_comments_and_echo_do_not_override_active_config(self):
  parsed=parse_launch('export HEAD_MODE=traj_only\necho "HEAD_MODE=$HEAD_MODE"\n# export EMBED_DIMS=256\npython train.py \\\n+--py-config config/model.py --load-from "$LPAI_INPUT_MODEL_8/epoch_83.pth"\n# --resume-from /wrong/epoch_76.pth\n')
  self.assertEqual(parsed['environment'],{'HEAD_MODE':'traj_only'})
  self.assertEqual(parsed['flags']['load-from'],'$LPAI_INPUT_MODEL_8/epoch_83.pth')
  self.assertNotIn('resume-from',parsed['flags'])
 def test_literal_shell_content_is_only_parsed_and_secrets_not_persisted(self):
  parsed=parse_launch('export API_KEY=hidden\nexport HEAD_MODE=gs_only # note\npython eval.py --py-config=config.py --traj-resume "path with spaces.pth"')
  self.assertNotIn('API_KEY',parsed['environment'])
  self.assertEqual(parsed['flags']['traj-resume'],'path with spaces.pth')
  self.assertEqual(parsed['flags']['py-config'],'config.py')

if __name__=='__main__':unittest.main()
