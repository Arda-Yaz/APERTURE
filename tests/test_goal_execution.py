import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
import isolation_support as safety
safety.install_guards()
import experience
import goals
import goal_execution as execution
import llm
import main


def response(content='', calls=()):
    return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[
        SimpleNamespace(function=SimpleNamespace(name=name, arguments=args))
        for name, args in calls]))


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        directory = safety.TestTemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = goals.GoalStore(self.root / 'goals.db')
        self.events = experience.ExperienceStore(self.root / 'experience.db')
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for module, name, value in [(goals, '_STORE', self.store),
                                    (experience, '_STORE', self.events),
                                    (experience, '_CURRENT_EPISODE_ID', None),
                                    (llm, 'ENABLE_DYNAMIC_SELF', False),
                                    (llm, 'ENABLE_RELATIONSHIP', False)]:
            self.stack.enter_context(patch.object(module, name, value))
        self.stack.enter_context(patch.object(llm, 'inject_runtime_context', side_effect=lambda messages, **kw: list(messages)))
        self.stack.enter_context(patch.object(llm, 'maybe_reflect', return_value=None))
        self.stack.enter_context(patch.object(llm, 'is_task_complete', return_value=True))
        self.permission = self.stack.enter_context(patch.object(llm, 'check_permission', return_value=True))
        self.output = self.stack.enter_context(patch('builtins.print'))
        self.goal_id = self.store.create('Inspect files and produce report')['id']
        self.read = Mock(return_value='observed content')
        self.write = Mock(return_value='File written')
        self.stack.enter_context(patch.dict(llm.TOOL_MAP, {'read_file': self.read, 'write_file': self.write}))

    def run_session(self, cycles, responses):
        messages = [{'role': 'user', 'content': f'/goal run {cycles}'}]
        with patch.object(llm, 'ollama_chat', side_effect=responses) as actor:
            result = execution.run_goal(messages, persistent_goal_id=self.goal_id, cycles=cycles)
        return result, actor, messages

    def read_cycle(self, path):
        return [response(calls=[('read_file', {'path': path})]), response('Cycle done')]

    def test_one_cycle_and_goal_remains_active(self):
        result, actor, _ = self.run_session(1, self.read_cycle('first'))
        self.assertEqual(actor.call_count, 2)
        self.assertEqual(self.read.call_count, 1)
        self.assertIn('cycle budget exhausted', result)
        self.assertEqual(self.store.get(self.goal_id)['status'], 'active')

    def test_three_cycles_grounded_context_and_single_authorization(self):
        captured = []
        responses = iter(self.read_cycle('first') + self.read_cycle('second') + self.read_cycle('third'))
        def actor(**kwargs):
            captured.append([dict(m) for m in kwargs['messages'] if isinstance(m, dict)])
            return next(responses)
        result, calls, messages = self.run_session(3, actor)
        self.assertEqual(calls.call_count, 6)
        self.assertEqual(self.read.call_count, 3)
        self.assertIn('cycle budget exhausted', result)
        second_context = '\n'.join(m['content'] for m in captured[2] if m['role'] == 'system')
        self.assertIn('observed content', second_context)
        self.assertIn('Inspect files and produce report', second_context)
        self.assertIn('unexecuted plans', second_context)
        self.assertEqual(sum(m['role'] == 'user' for m in messages), 1)
        events = experience.get_episode_events(experience._CURRENT_EPISODE_ID, limit=100)
        user = [e for e in events if e['event_type'] == 'user_message']
        self.assertEqual(len(user), 1)
        self.assertEqual(user[0]['content'], '/goal run 3')
        self.assertTrue(all(e['actor'] == 'system' and e['event_class'] == 'system' for e in events if e['event_type'].startswith('goal_execution_')))

    def test_invalid_budgets_and_inactive_goals(self):
        for value in ('', '0', '-1', '4', '1.5', '1 extra', '+1', '٣'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                goals.handle_goal_command('/goal run ' + value)
        for value in (0, 4, True, '3'):
            with self.subTest(api=value), self.assertRaises(ValueError):
                execution.run_goal([], persistent_goal_id=self.goal_id, cycles=value)
        goals.handle_goal_command('/goal pause')
        with self.assertRaises(ValueError):
            goals.handle_goal_command('/goal run 1')
        self.assertEqual(goals.handle_goal_command('/goal resume')[1], self.goal_id)
        goals.handle_goal_command('/goal complete')
        with self.assertRaises(ValueError):
            goals.handle_goal_command('/goal run 1')

    def test_multi_call_response_stops_before_thirteenth_dispatch(self):
        result, actor, _ = self.run_session(3, [response(calls=[('read_file', {'path': str(i)}) for i in range(20)])])
        self.assertEqual(self.read.call_count, 12)
        self.assertEqual(self.permission.call_count, 12)
        self.assertEqual(len(self.store.actions(self.goal_id)), 12)
        self.assertEqual(actor.call_count, 1)
        self.assertIn('tool attempt budget exhausted', result)

    def test_budget_shared_across_cycles_including_checkpoint_and_memory(self):
        search = Mock(return_value='memory observations')
        with patch.dict(llm.TOOL_MAP, {'search_memory': search}):
            calls = [('checkpoint_goal', {'progress': 'Claim', 'next_action': 'Plan'})] * 5
            calls += [('search_memory', {'query': 'report'})] * 5
            result, actor, _ = self.run_session(3, self.read_cycle('one') + [response(calls=calls + [('read_file', {'path': 'two'}), ('read_file', {'path': 'three'})])])
        self.assertEqual(len(self.store.actions(self.goal_id)), 12)
        self.assertEqual(self.read.call_count, 2)
        self.assertEqual(search.call_count, 5)
        self.assertIn('tool attempt budget exhausted', result)

    def test_checkpoint_or_final_claim_alone_is_not_progress(self):
        for calls in ([], [('checkpoint_goal', {'progress': 'Everything finished', 'next_action': 'verify'})],
                      [('recall_recent_activity', {})]):
            with self.subTest(calls=calls), patch.dict(llm.TOOL_MAP, {'recall_recent_activity': lambda: 'old events'}):
                sequence = ([response(calls=calls)] if calls else []) + [response('Everything done')]
                result, actor, _ = self.run_session(3, sequence)
                self.assertIn('no new meaningful tool observations', result)
                self.assertEqual(actor.call_count, len(sequence))
                self.assertEqual(self.store.get(self.goal_id)['status'], 'active')

    def test_repeated_identical_observation_stops(self):
        result, actor, _ = self.run_session(3, self.read_cycle('same') * 2)
        self.assertEqual(actor.call_count, 4)
        self.assertIn('no new meaningful tool observations', result)

    def test_permission_denial_stops_same_batch(self):
        self.permission.return_value = False
        result, actor, _ = self.run_session(3, [response(calls=[('write_file', {'path': 'first', 'content': 'x'}), ('read_file', {'path': 'second'})])])
        self.assertIn('permission denied', result)
        self.write.assert_not_called()
        self.read.assert_not_called()
        self.assertEqual(actor.call_count, 1)
        self.assertEqual(self.store.actions(self.goal_id)[0]['status'], 'permission_denied')

    def test_tool_errors_stop_same_batch_and_uncertain_mutation_blocks_restart(self):
        self.write.return_value = 'WRITE_ERROR: partial effects'
        result, actor, _ = self.run_session(3, [response(calls=[('write_file', {'path': 'first', 'content': 'x'}), ('read_file', {'path': 'second'})])])
        self.assertIn('tool error', result)
        self.assertEqual(actor.call_count, 1)
        self.read.assert_not_called()
        with patch.object(goals, '_STORE', goals.GoalStore(self.store.db_path)):
            result, actor, _ = self.run_session(3, [])
        self.assertIn('explicit reconciliation required', result)
        actor.assert_not_called()

    def test_successful_mutation_is_not_replayed_after_restart(self):
        args = {'path': 'report', 'content': 'report'}
        self.run_session(1, [response(calls=[('write_file', args), ('checkpoint_goal', {'progress': 'Report created', 'next_action': 'verify'})]), response('Written')])
        with patch.object(goals, '_STORE', goals.GoalStore(self.store.db_path)):
            result, actor, _ = self.run_session(3, [response(calls=[('write_file', args), ('read_file', {'path': 'report'})])])
        self.assertEqual(self.write.call_count, 1)
        self.read.assert_not_called()
        self.assertIn('action blocked', result)
        self.assertEqual(self.store.get(self.goal_id)['next_action'], 'verify')
        self.assertEqual(actor.call_count, 1)

    def test_interrupt_preserves_pending_intent_and_no_automatic_retry(self):
        self.write.side_effect = KeyboardInterrupt
        result, actor, _ = self.run_session(3, [response(calls=[('write_file', {'path': 'report', 'content': 'x'}), ('read_file', {'path': 'other'})])])
        self.assertIn('user interrupted', result)
        action = self.store.actions(self.goal_id)[0]
        self.assertEqual(action['status'], 'pending')
        self.assertEqual(self.store.get(self.goal_id)['status'], 'active')
        result, actor, _ = self.run_session(3, [])
        actor.assert_not_called()
        self.assertIn('explicit reconciliation required', result)

    def test_result_recording_crash_leaves_pending_intent(self):
        with patch.object(llm, 'record_tool_result', side_effect=RuntimeError('disk full')):
            result, actor, _ = self.run_session(3, [response(calls=[('write_file', {'path': 'report', 'content': 'x'})])])
        self.assertIn('execution error', result)
        self.assertEqual(self.write.call_count, 1)
        self.assertEqual(self.store.actions(self.goal_id)[0]['status'], 'pending')
        result, actor, _ = self.run_session(3, [])
        actor.assert_not_called()

    def test_goal_deactivation_blocks_next_call_in_same_batch(self):
        def deactivate(**kwargs):
            self.store.transition(self.goal_id, 'paused')
            return 'observed'
        self.read.side_effect = deactivate
        result, actor, _ = self.run_session(3, [response(calls=[('read_file', {'path': 'first'}), ('write_file', {'path': 'report', 'content': 'x'})])])
        self.assertIn('goal is inactive', result)
        self.write.assert_not_called()
        self.assertEqual(self.store.actions(self.goal_id)[0]['status'], 'ok')

    def test_per_cycle_model_step_budget_stops_session(self):
        with patch.object(llm, 'MAX_STEPS', 2):
            result, actor, _ = self.run_session(3, [response(calls=[('read_file', {'path': 'one'})]), response(calls=[('read_file', {'path': 'two'})])])
        self.assertIn('per-cycle model step budget exhausted', result)
        self.assertEqual(actor.call_count, 2)

    def test_cli_routes_only_explicit_run_and_startup_never_executes(self):
        with patch('builtins.input', side_effect=['hello', '/goal run 2', 'quit']), patch.object(main, 'chat', return_value='hello') as ordinary, patch.object(main, 'run_goal', return_value='stop') as bounded:
            main.main()
        ordinary.assert_called_once()
        self.assertEqual(ordinary.call_args.kwargs, {'persistent_goal_id': None})
        bounded.assert_called_once()
        self.assertEqual(bounded.call_args.kwargs, {'persistent_goal_id': self.goal_id, 'cycles': 2})
        with patch('builtins.input', return_value='quit'), patch.object(main, 'run_goal') as bounded:
            main.main()
        bounded.assert_not_called()

    def test_normal_conversation_runs_single_actor_without_session(self):
        with patch.object(llm, 'ollama_chat', return_value=response('hello')) as actor, patch.object(llm, 'get_store') as store:
            llm.chat([{'role': 'user', 'content': 'hello'}])
        actor.assert_called_once()
        store.assert_not_called()


if __name__ == '__main__':
    unittest.main()
