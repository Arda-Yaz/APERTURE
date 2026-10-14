import sys
import json
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
from task_controller import Assessment
import task_controller
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
        self.judge = self.stack.enter_context(patch.object(llm, 'assess_task', return_value=Assessment('incomplete', 'More work remains.', (), ('Finish report.',))))
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
        self.judge.side_effect = [Assessment('incomplete', 'Inspect more.', (), ('More files.',)),
                                  Assessment('incomplete', 'Inspect more.', (), ('More files.',)),
                                  Assessment('satisfied', 'Report supported.', ('fixture',))]
        captured = []
        responses = iter(self.read_cycle('first') + self.read_cycle('second') + self.read_cycle('third'))
        def actor(**kwargs):
            captured.append([dict(m) for m in kwargs['messages'] if isinstance(m, dict)])
            return next(responses)
        result, calls, messages = self.run_session(3, actor)
        self.assertEqual(calls.call_count, 6)
        self.assertEqual(self.read.call_count, 3)
        self.assertIn('objective assessed satisfied', result)
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

    def actual_controller(self, statuses):
        self.judge.side_effect = task_controller.assess_task
        queue = iter(statuses)
        def infer(**kwargs):
            payload = json.loads(kwargs['messages'][1]['content'])
            refs = [entry['evidence']['result_event_id'] for entry in payload['evidence']['entries']
                    if entry['category'] == 'successful_observations']
            status = next(queue)
            return response(json.dumps({'status': status, 'rationale': 'Assessment over supplied observations.',
                            'evidence_refs': refs if status == 'satisfied' else [],
                            'missing_requirements': ['Finish report.'] if status == 'incomplete' else []}))
        return patch.object(task_controller, 'ollama_chat', side_effect=infer)

    def test_supported_satisfaction_stops_first_cycle_and_preserves_answer(self):
        with self.actual_controller(['satisfied']) as controller:
            result, actor, _ = self.run_session(3, self.read_cycle('first'))
        self.assertIn('Cycle done', result)
        self.assertIn('objective assessed satisfied', result)
        self.assertEqual(actor.call_count, 2)
        self.assertEqual(controller.call_count, 1)
        self.assertEqual(self.read.call_count, 1)
        self.assertEqual(self.store.get(self.goal_id)['status'], 'active')
        events = experience.get_episode_events(experience._CURRENT_EPISODE_ID, limit=100)
        assessments = [e for e in events if e['event_type'] == 'task_assessment']
        self.assertEqual(len(assessments), 1)
        self.assertEqual(assessments[0]['event_class'], 'derived')
        self.assertTrue(assessments[0]['metadata']['assessment']['evidence_refs'])
        # Explicit authorization creates a fresh session, not a restored verdict.
        with self.actual_controller(['insufficient_evidence']) as controller:
            result, actor, _ = self.run_session(3, [response('Claim after restart')])
        self.assertIn('insufficient_evidence', result)
        self.assertEqual(controller.call_count, 1)

    def test_later_no_action_candidate_uses_retained_evidence(self):
        with self.actual_controller(['incomplete', 'satisfied']) as controller:
            result, actor, _ = self.run_session(3, self.read_cycle('first') + [response('Supported report')])
        self.assertIn('Supported report', result)
        self.assertIn('objective assessed satisfied', result)
        self.assertEqual(actor.call_count, 3)
        self.assertEqual(self.read.call_count, 1)
        self.assertEqual(controller.call_count, 2)
        inputs = [json.loads(call.kwargs['messages'][1]['content']) for call in controller.call_args_list]
        self.assertEqual(inputs[0]['evidence'], inputs[1]['evidence'])

    def test_unsupported_claim_checkpoint_and_failed_inference_stop(self):
        for calls in [[], [('checkpoint_goal', {'progress': 'Everything done', 'next_action': ''})]]:
            with self.subTest(calls=calls), self.actual_controller(['satisfied']) as controller:
                sequence = ([response(calls=calls)] if calls else []) + [response('All done')]
                result, actor, _ = self.run_session(3, sequence)
                self.assertIn('task assessment invalid', result)
                self.assertEqual(actor.call_count, len(sequence))
                self.assertEqual(controller.call_count, 1)
        self.judge.side_effect = task_controller.assess_task
        with patch.object(task_controller, 'ollama_chat', side_effect=RuntimeError('offline')) as controller:
            result, actor, _ = self.run_session(3, self.read_cycle('failure'))
        self.assertIn('task assessment unavailable', result)
        self.assertEqual(controller.call_count, 1)
        self.assertEqual(actor.call_count, 2)

    def test_safety_changes_during_assessment_override_satisfaction(self):
        def assess(*args, **kwargs):
            self.store.transition(self.goal_id, 'paused')
            return Assessment('satisfied', 'Observed.', tuple(args[2].eligible_refs))
        self.judge.side_effect = assess
        result, actor, _ = self.run_session(3, self.read_cycle('first'))
        self.assertIn('goal is inactive', result)
        self.assertNotIn('Stopped: objective assessed satisfied', result)
        self.assertEqual(actor.call_count, 2)

    def test_finalization_interrupt_or_recording_failure_overrides_satisfaction(self):
        for name, error, reason in [('maybe_reflect', KeyboardInterrupt(), 'user interrupted'),
                                    ('record_assistant_message', RuntimeError('disk full'), 'execution error')]:
            with self.subTest(name=name), self.actual_controller(['satisfied']) as controller, \
                    patch.object(llm, name, side_effect=error):
                result, actor, _ = self.run_session(3, self.read_cycle(name))
            self.assertIn('Cycle done', result)
            self.assertIn(reason, result)
            self.assertNotIn('Stopped: objective assessed satisfied', result)
            self.assertEqual(actor.call_count, 2)
            self.assertEqual(controller.call_count, 1)
            self.assertEqual(self.store.get(self.goal_id)['status'], 'active')
            events = experience.get_episode_events(experience._CURRENT_EPISODE_ID, limit=100)
            stops = [event for event in events if event['event_type'] == 'goal_execution_stopped']
            self.assertIn(reason, stops[-1]['content'])
            self.assertEqual(stops[-1]['event_class'], 'system')

    def test_finalization_safety_change_overrides_satisfaction(self):
        def deactivate(*args, **kwargs):
            self.store.transition(self.goal_id, 'paused')
        with self.actual_controller(['satisfied']), patch.object(llm, 'maybe_reflect', side_effect=deactivate):
            result, actor, _ = self.run_session(3, self.read_cycle('first'))
        self.assertIn('goal is inactive', result)
        self.assertIn('Cycle done', result)
        self.assertNotIn('Stopped: objective assessed satisfied', result)
        self.assertEqual(actor.call_count, 2)

    def test_unchanged_candidate_evidence_is_cached(self):
        with self.actual_controller(['incomplete']) as controller:
            result, actor, _ = self.run_session(3, self.read_cycle('first') + [response('Cycle done')])
        self.assertEqual(controller.call_count, 1)
        self.assertEqual(actor.call_count, 3)
        self.assertIn('no new meaningful tool observations', result)
        events = experience.get_episode_events(experience._CURRENT_EPISODE_ID, limit=100)
        assessments = [e for e in events if e['event_type'] == 'task_assessment']
        self.assertEqual([e['metadata']['cached'] for e in assessments], [False, True])

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
        self.judge.assert_not_called()

    def test_real_workspace_tools_e2e_and_cross_cycle_controller_evidence(self):
        fixture = self.root / 'project'
        fixture.mkdir()
        (fixture / 'main.py').write_text('from persistence import Store\nStore()\n', encoding='utf-8')
        (fixture / 'persistence.py').write_text('import sqlite3\nclass Store:\n    pass\n', encoding='utf-8')
        root = str(fixture)
        captures, controller = [], []
        responses = iter([
            response(calls=[('discover_workspace', {'root': root, 'limit': 1})]), response('Found main.py.'),
            response(calls=[('search_workspace', {'root': root, 'query': 'class Store'})]), response('Store is in persistence.py:2.'),
            response(calls=[('read_source', {'root': root, 'path': 'persistence.py', 'start_line': 1, 'line_count': 3}),
                            ('read_source', {'root': root, 'path': 'main.py', 'line_count': 2})]),
            response('main.py is the entry point; persistence.py:1 imports sqlite3 and persistence.py:2 defines Store.')])
        def actor(**kwargs):
            captures.append([dict(m) for m in kwargs['messages'] if isinstance(m, dict)])
            return next(responses)
        def judge(goal, answer, evidence, **kwargs):
            kwargs['evidence'] = evidence
            controller.append(json.dumps(kwargs['evidence'].payload()))
            return Assessment('satisfied' if len(controller) == 3 else 'incomplete', 'Evidence assessed.',
                              tuple(kwargs['evidence'].eligible_refs) if len(controller) == 3 else (),
                              () if len(controller) == 3 else ('Inspect remaining files.',))
        with patch.object(llm, 'assess_task', side_effect=judge):
            result, actor_mock, _ = self.run_session(3, actor)
        self.assertEqual(actor_mock.call_count, 6)
        self.assertEqual(len(self.store.actions(self.goal_id)), 4)
        self.assertEqual(self.store.get(self.goal_id)['status'], 'active')
        self.assertIn('objective assessed satisfied', result)
        later = '\n'.join(message['content'] for message in captures[4] if message['role'] == 'system')
        for value in ['main.py', 'persistence.py', 'continuation', 'call_event_id', 'result_event_id', json.dumps(root)[1:-1]]:
            self.assertIn(value, later)
            self.assertIn(value, controller[-1])
        self.assertIn('requested_start', controller[-1])
        self.assertIn('from persistence import Store', controller[-1])
        self.assertIn('Store()', controller[-1])
        for tool, target in self.permission.call_args_list:
            self.assertIn(root, tool[1])
        with patch.object(goals, '_STORE', goals.GoalStore(self.store.db_path)):
            summary = goals.goal_summary(self.goal_id)
        self.assertIn('class Store', summary)
        self.assertIn('requested_start', summary)
        self.assertIn('SANITIZED WORKSPACE EVIDENCE', summary)

    def test_workspace_error_and_denial_stop_same_batch_count_attempts(self):
        for denial in [False, True]:
            with self.subTest(denial=denial):
                self.permission.return_value = not denial
                previous = len(self.store.actions(self.goal_id))
                result, actor, _ = self.run_session(3, [response(calls=[
                    ('read_source', {'root': str(self.root), 'path': 'missing.py'}),
                    ('read_file', {'path': 'should-not-run'})])])
                self.read.assert_not_called()
                self.assertEqual(actor.call_count, 1)
                self.assertEqual(len(self.store.actions(self.goal_id)), previous + 1)
                self.assertIn('permission denied' if denial else 'tool error', result)
                self.assertEqual(self.store.actions(self.goal_id)[-1]['status'], 'permission_denied' if denial else 'error')

    def test_session_uses_sanitized_events_and_bounds_retained_evidence(self):
        session = execution.ExecutionSession(self.goal_id, 3, 'fixture')
        for index in range(12):
            call = {'id': f'call{index}', 'metadata': {'tool_name': 'read_source',
                    'arguments': {'root': str(self.root), 'path': 'a.py', 'secret': '[REDACTED]'}}}
            result = {'id': f'result{index}', 'content': json.dumps({'tool': 'read_source',
                      'root': str(self.root), 'path': 'a.py', 'status': 'ok',
                      'continuation': 'next-position', 'range': {'returned_start': index},
                      'coverage': {'complete': False}, 'evidence': [{'text': 'x' * 15000}]}),
                      'metadata': {'status': 'ok'}}
            session.observe('read_source', {'path': str(index), 'secret': 'raw-secret'},
                            'raw' + str(index), 'ok', call_event=call, result_event=result)
        context = session.context()
        self.assertLessEqual(sum(map(len, session.observations)), 16000)
        self.assertNotIn('raw-secret', context)
        self.assertIn('metadata_omitted', context)
        self.assertIn('next-position', context)
        self.assertIn('call11', context)
        self.assertIn('result11', context)
        self.assertIn('evidence_excerpt_truncated', context)
        self.assertEqual(len(session.seen), 12)

    def test_large_write_arguments_remain_referenced_without_unbounded_context(self):
        session = execution.ExecutionSession(self.goal_id, 3, 'fixture')
        call = {'id': 'large_call', 'metadata': {'tool_name': 'write_file',
                'arguments': {'path': 'report.txt', 'content': 'x' * 20000}}}
        result = {'id': 'large_result', 'content': 'File written', 'metadata': {'status': 'ok'}}
        session.observe('write_file', {'path': 'report.txt'}, 'File written', 'ok',
                        call_event=call, result_event=result)
        self.assertLessEqual(len(session.observations[0]), 10000)
        self.assertLessEqual(sum(map(len, session.observations)), 16000)
        decoded = json.loads(session.observations[0])
        self.assertEqual(decoded['arguments'], {'path': 'report.txt'})
        self.assertEqual(decoded['call_event_id'], 'large_call')
        self.assertEqual(decoded['result_event_id'], 'large_result')
        self.assertEqual(decoded['status'], 'ok')
        self.assertTrue(decoded['arguments_fingerprint'])
        self.assertTrue(decoded['metadata_omitted'])

    def test_large_inspection_metadata_bound_preserves_usable_cursor(self):
        from workspace import evidence_context
        call = {'id': 'inspection_call', 'metadata': {'tool_name': 'search_workspace',
                'arguments': {'root': str(self.root), 'path': '.', 'query': 'symbol'}}}
        cursor = json.dumps({'root': str(self.root), 'position': 'x' * 2500})
        result = {'id': 'inspection_result', 'content': json.dumps({
                  'root': str(self.root), 'path': '.', 'query': 'symbol',
                  'status': 'ok', 'continuation': cursor, 'truncated': True,
                  'coverage': {'complete': False, 'skipped': ['huge' * 10000]},
                  'evidence': [{'path': 'source.py', 'line': 42, 'text': 'symbol' * 10000}]}),
                  'metadata': {'status': 'ok'}}
        context = evidence_context(call, result, max_chars=10000)
        self.assertLessEqual(len(context), 10000)
        decoded = json.loads(context)
        self.assertEqual(decoded['inspection']['continuation'], cursor)
        self.assertEqual(decoded['inspection']['root'], str(self.root))
        self.assertFalse(decoded['inspection']['coverage']['complete'])
        self.assertTrue(decoded['evidence_excerpt_truncated'])
        self.assertTrue(decoded['metadata_omitted'])


if __name__ == '__main__':
    unittest.main()
