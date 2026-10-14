import json
import unittest
from unittest.mock import Mock, patch
import test_goal_execution as fixtures
import experience
import llm
import task_controller
import workspace as ws
import goals


response = fixtures.response


class WorkspaceWriteExecutionTests(unittest.TestCase):
    setUp = fixtures.ExecutionTests.setUp
    run_session = fixtures.ExecutionTests.run_session
    def args(self, path='report.txt', content='verified report'):
        return {'root': str(self.root), 'path': path, 'content': content}

    def test_pending_commits_before_write_and_success_blocks_replay(self):
        real = llm.TOOL_MAP['create_text']
        def observed(**args):
            pending = self.store.actions(self.goal_id)[-1]
            self.assertEqual(pending['status'], 'pending')
            self.assertFalse((self.root / args['path']).exists())
            return real(**args)
        args = self.args()
        with patch.dict(llm.TOOL_MAP, {'create_text': observed}):
            self.run_session(1, [response(calls=[('create_text', args)]), response('report created')])
        self.assertEqual(self.store.actions(self.goal_id)[0]['status'], 'ok')
        self.assertIn('already succeeded', self.store.replay_block(self.goal_id, 'create_text', args))
        summary = goals.goal_summary(self.goal_id)
        self.assertIn('exact_bytes', summary)
        self.assertIn('fingerprint', summary)
        self.assertIn('result_event_id', summary)

    def test_denial_without_effects_and_visible_permission_target(self):
        self.permission.return_value = False
        args = self.args()
        result, actor, _ = self.run_session(3, [response(calls=[('create_text', args)])])
        self.assertFalse((self.root / 'report.txt').exists())
        self.assertEqual(actor.call_count, 1)
        self.assertIn('permission denied', result)
        self.permission.assert_called_with('create_text', str(self.root) + ' :: report.txt')

    def test_conflict_stops_ordinary_resume_without_legacy_fallback(self):
        target = self.root / 'report.txt'
        target.write_bytes(b'existing')
        with patch.object(llm, 'ollama_chat', return_value=response(calls=[('create_text', self.args()), ('write_file', {'path': 'report.txt', 'content': 'bad'})])) as actor:
            result = llm.chat([{'role': 'user', 'content': '/goal resume'}], persistent_goal_id=self.goal_id)
        self.assertIn('conflict', result)
        self.write.assert_not_called()
        self.assertEqual(actor.call_count, 1)
        self.assertEqual(target.read_bytes(), b'existing')
        self.assertIsNone(self.store.replay_block(self.goal_id, 'create_text', self.args(content='retry explicitly')))

    def test_no_persistent_goal_rejects_before_dispatch(self):
        writer = Mock()
        with patch.dict(llm.TOOL_MAP, {'create_text': writer}), patch.object(llm, 'ollama_chat', return_value=response(calls=[('create_text', self.args())])):
            result = llm.chat([{'role': 'user', 'content': 'Create report'}])
        writer.assert_not_called()
        self.assertIn('rejected', result)

    def test_result_recording_interruption_preserves_pending_and_restart_block(self):
        with patch.object(llm, 'record_tool_result', side_effect=RuntimeError('recording interrupted')):
            result, _, _ = self.run_session(1, [response(calls=[('create_text', self.args())])])
        self.assertTrue((self.root / 'report.txt').exists())
        self.assertEqual(self.store.actions(self.goal_id)[0]['status'], 'pending')
        self.assertIn('execution error', result)
        self.assertIn('uncertain effects', self.store.replay_block(self.goal_id, 'create_text', self.args(path='different.txt')))

    def test_postpublication_error_requires_reconciliation(self):
        real = ws._publish_text
        def corrupt(temp, target, update):
            real(temp, target, update)
            target.write_bytes(b'changed')
        with patch.object(ws, '_publish_text', side_effect=corrupt):
            self.run_session(1, [response(calls=[('create_text', self.args())])])
        self.assertEqual(self.store.actions(self.goal_id)[0]['status'], 'error')
        self.assertIn('uncertain effects', self.store.replay_block(self.goal_id, 'create_text', self.args(path='different.txt')))

    def test_malformed_write_shape_is_no_effect_rejection(self):
        target = self.root / 'config.txt'
        target.write_bytes(b'unchanged')
        invalid = [self.args(path='config.txt'), {**self.args(path='config.txt'), 'extra': 'bad'}]
        for args in invalid:
            with patch.object(llm, 'ollama_chat', return_value=response(calls=[('update_text', args)])):
                result = llm.chat([{'role': 'user', 'content': '/goal resume'}], persistent_goal_id=self.goal_id)
            self.assertIn('rejected', result)
            self.assertEqual(self.store.actions(self.goal_id)[-1]['status'], 'rejected')
            self.assertIsNone(self.store.replay_block(self.goal_id, 'update_text', args))
            self.assertEqual(target.read_bytes(), b'unchanged')

    def test_finish_recording_interruption_is_pending_after_reopen(self):
        with patch.object(self.store, 'finish_action', side_effect=RuntimeError('finish interrupted')):
            self.run_session(1, [response(calls=[('create_text', self.args())])])
        reopened = goals.GoalStore(self.root / 'goals.db')
        action = reopened.actions(self.goal_id)[0]
        self.assertEqual(action['status'], 'pending')
        self.assertTrue((self.root / 'report.txt').exists())
        self.assertIn('uncertain effects', reopened.replay_block(self.goal_id, 'create_text', self.args(path='different.txt')))
        events = experience.get_episode_events(experience._CURRENT_EPISODE_ID, limit=100)
        results = [event for event in events if event['event_type'] == 'tool_result']
        self.assertEqual(results[0]['metadata']['status'], 'ok')
        self.assertEqual(json.loads(results[0]['content'])['status'], 'verified')

    def test_begin_recording_failure_never_dispatches(self):
        writer = Mock()
        with patch.dict(llm.TOOL_MAP, {'create_text': writer}), patch.object(self.store, 'begin_action', side_effect=RuntimeError('begin failed')):
            self.run_session(1, [response(calls=[('create_text', self.args())])])
        writer.assert_not_called()
        self.assertFalse((self.root / 'report.txt').exists())
        self.assertEqual(self.store.actions(self.goal_id), [])

    def test_inspect_update_assessment_uses_actual_verification(self):
        target = self.root / 'config.txt'
        target.write_bytes(b'old\r\nkeep\n')
        args = {'root': str(self.root), 'path': 'config.txt'}
        expected = json.loads(ws.snapshot_text(**args))['fingerprint']
        update_args = {**args, 'content': 'new\r\nkeep\n', 'expected_fingerprint': expected}
        self.judge.side_effect = task_controller.assess_task
        captured = []
        def assess(**kwargs):
            payload = json.loads(kwargs['messages'][1]['content'])
            captured.append(payload)
            entries = payload['evidence']['entries']
            verified = [entry for entry in entries if entry['category'] == 'successful_observations' and entry['evidence']['tool'] == 'update_text']
            self.assertEqual(len(verified), 1)
            evidence = verified[0]['evidence']
            self.assertEqual(evidence['inspection']['verification'], {'exact_bytes': True, 'readback': True})
            self.assertEqual(evidence['inspection']['fingerprint'], json.loads(ws.snapshot_text(**args))['fingerprint'])
            return response(json.dumps({'status': 'satisfied', 'rationale': 'Observed readback verifies requested bytes.', 'evidence_refs': [evidence['result_event_id']], 'missing_requirements': []}))
        with patch.object(task_controller, 'ollama_chat', side_effect=assess):
            result, _, _ = self.run_session(3, [response(calls=[('snapshot_text', args)]), response(calls=[('update_text', update_args)]), response('Configuration updated')])
        self.assertIn('objective assessed satisfied', result)
        self.assertEqual(target.read_bytes(), b'new\r\nkeep\n')
        self.assertEqual(self.store.get(self.goal_id)['status'], 'active')
        self.assertEqual(len(captured), 1)
