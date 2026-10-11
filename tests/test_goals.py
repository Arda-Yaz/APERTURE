import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
import isolation_support as safety
safety.install_guards()

import experience
import goals
import llm
import main


def response(content='', calls=()):
    return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[
        SimpleNamespace(function=SimpleNamespace(name=name, arguments=args))
        for name, args in calls
    ]))


class GoalTests(unittest.TestCase):
    def setUp(self):
        directory = safety.TestTemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / 'goals.db'
        self.store = goals.GoalStore(self.path)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(goals, '_STORE', self.store))
        self.stack.enter_context(patch.object(experience, '_STORE', experience.ExperienceStore(self.root / 'experience.db')))
        self.stack.enter_context(patch.object(experience, '_CURRENT_EPISODE_ID', None))
        self.stack.enter_context(patch.object(llm, 'inject_runtime_context', side_effect=lambda messages, **kw: list(messages)))
        self.stack.enter_context(patch.object(llm, 'maybe_reflect', return_value=None))
        self.stack.enter_context(patch.object(llm, 'ENABLE_DYNAMIC_SELF', False))
        self.stack.enter_context(patch.object(llm, 'ENABLE_RELATIONSHIP', False))
        self.stack.enter_context(patch.object(llm, 'is_task_complete', return_value=True))
        self.permission = self.stack.enter_context(patch.object(llm, 'check_permission', return_value=True))
        self.stack.enter_context(patch('builtins.print'))

    def start(self):
        goals.handle_goal_command('/goal start Write first.txt then second.txt')
        return self.store.unfinished()['id']

    def turn(self, goal_id, responses, instruction='/goal resume'):
        with patch.object(llm, 'ollama_chat', side_effect=responses) as actor:
            result = llm.chat([{'role':'user', 'content':instruction}], persistent_goal_id=goal_id)
        return result, actor

    def test_create_restart_checkpoint_and_one_unfinished(self):
        goal_id = self.start()
        self.store.checkpoint(goal_id, 'First file reported done', 'Create second file')
        reopened = goals.GoalStore(self.path)
        self.assertEqual(reopened.unfinished()['id'], goal_id)
        self.assertEqual(reopened.get(goal_id)['next_action'], 'Create second file')
        with self.assertRaises(ValueError):
            reopened.create('Another goal')

    def test_pause_resume_and_terminal_history(self):
        for terminal in ('complete', 'cancel'):
            with self.subTest(terminal=terminal):
                goal_id = self.start()
                goals.handle_goal_command('/goal pause')
                with self.assertRaises(ValueError):
                    self.store.require_active(goal_id)
                self.assertEqual(goals.handle_goal_command('/goal resume')[1], goal_id)
                goals.handle_goal_command('/goal ' + terminal)
                self.assertIsNone(self.store.unfinished())
                with self.assertRaises(ValueError):
                    self.store.transition(goal_id, 'active')
                with patch.object(llm, 'ollama_chat') as actor, self.assertRaises(ValueError):
                    llm.chat([{'role':'user', 'content':'resume'}], persistent_goal_id=goal_id)
                actor.assert_not_called()
                self.assertIsNotNone(self.store.get(goal_id))

    def test_start_and_startup_do_not_execute(self):
        with patch.object(llm, 'ollama_chat') as actor:
            self.start()
            with patch('builtins.input', return_value='quit'), patch.object(main, 'chat') as chat:
                main.main()
            chat.assert_not_called()
            actor.assert_not_called()

    def test_cli_explicit_resume_binds_goal_and_ordinary_chat_does_not(self):
        goal_id = self.start()
        with patch('builtins.input', side_effect=['hello', '/goal pause', '/goal resume Remaining step', 'quit']), patch.object(main, 'chat', return_value='ok') as chat:
            main.main()
        self.assertEqual(chat.call_args_list[0].kwargs, {'persistent_goal_id':None})
        self.assertEqual(chat.call_args_list[1].kwargs, {'persistent_goal_id':goal_id})

    def test_ordinary_chat_never_touches_goal_store_and_tool_is_not_exposed(self):
        with patch.object(llm, 'get_store') as store:
            _, actor = self.turn(None, [response('Hello')], instruction='Hello')
            store.assert_not_called()
        self.assertIsNone(self.store.unfinished())
        self.assertNotIn('checkpoint_goal', [tool.__name__ for tool in actor.call_args.kwargs['tools']])
        with patch.object(llm, 'get_store') as store:
            self.turn(None, [response(calls=[('checkpoint_goal', {'progress':'done','next_action':''})]), response('blocked')])
            store.assert_not_called()

    def test_real_file_write_restart_resume_remaining_without_replay(self):
        goal_id = self.start()
        first = {'path':str(self.root / 'first.txt'), 'content':'first'}
        second = {'path':str(self.root / 'second.txt'), 'content':'second'}
        self.turn(goal_id, [response(calls=[('write_file', first), ('checkpoint_goal', {'progress':'First file written','next_action':'Create second file'})]), response('First step done')])
        self.assertEqual((self.root / 'first.txt').read_text(), 'first')
        self.assertEqual(self.store.get(goal_id)['status'], 'active')
        original_actions = self.store.actions(goal_id)
        actual = original_actions[0]
        self.assertEqual(actual['status'], 'ok')
        ids = [actual['call_event_id'], actual['result_event_id']]
        self.assertEqual(len(experience.get_events_by_ids(ids)), 2)
        self.store = goals.GoalStore(self.path)
        with patch.object(goals, '_STORE', self.store):
            _, actor = self.turn(goal_id, [response(calls=[('write_file', first), ('write_file', second)]), response('Finished')])
        self.assertEqual((self.root / 'second.txt').read_text(), 'second')
        self.assertEqual(self.permission.call_count, 2)
        self.assertEqual(self.store.actions(goal_id)[-2]['status'], 'blocked')
        self.assertIn('Create second file', actor.call_args.kwargs['messages'][1]['content'])
        self.assertEqual(self.store.get(goal_id)['status'], 'active')

    def test_crash_after_side_effect_before_result_is_durable_and_blocks_mutations(self):
        goal_id = self.start()
        args = {'path':str(self.root / 'first.txt'), 'content':'first'}
        with patch.object(llm, 'record_tool_result', side_effect=RuntimeError('crash')):
            with self.assertRaises(RuntimeError):
                self.turn(goal_id, [response(calls=[('write_file', args)])])
        self.assertTrue((self.root / 'first.txt').is_file())
        pending = goals.GoalStore(self.path).actions(goal_id)[0]
        self.assertEqual(pending['status'], 'pending')
        remaining = {'path':str(self.root / 'second.txt'), 'content':'second'}
        self.turn(goal_id, [response(calls=[('read_file', {'path':args['path']}), ('write_file', remaining)]), response('Waiting')])
        self.assertFalse((self.root / 'second.txt').exists())
        goals.handle_goal_command(f"/goal resolve {pending['id']} succeeded")
        self.turn(goal_id, [response(calls=[('write_file', args), ('write_file', remaining)]), response('done')])
        self.assertTrue((self.root / 'second.txt').exists())
        self.assertEqual(self.store.actions(goal_id)[0]['status'], 'pending')
        self.assertEqual(self.store.actions(goal_id)[0]['resolution'], 'succeeded')

    def test_interruption_before_tool_return_and_explicit_retry(self):
        goal_id = self.start()
        args = {'path':str(self.root / 'first.txt'), 'content':'first'}
        with patch.dict(llm.TOOL_MAP, {'write_file':lambda **kw: (_ for _ in ()).throw(KeyboardInterrupt())}):
            with self.assertRaises(KeyboardInterrupt):
                self.turn(goal_id, [response(calls=[('write_file', args)])])
        action = self.store.actions(goal_id)[0]
        self.assertEqual(action['status'], 'pending')
        goals.handle_goal_command(f"/goal resolve {action['id']} retry")
        self.turn(goal_id, [response(calls=[('write_file', args)]), response('done')])
        self.assertTrue((self.root / 'first.txt').exists())
        self.assertEqual(self.permission.call_count, 2)

    def test_failed_execution_and_permission_denial_are_not_success(self):
        goal_id = self.start()
        args = {'path':str(self.root / 'first.txt'), 'content':'first'}
        with patch.dict(llm.TOOL_MAP, {'write_file':lambda **kw: 'WRITE_ERROR: failed'}):
            self.turn(goal_id, [response(calls=[('write_file', args)]), response('done')])
        self.assertEqual(self.store.actions(goal_id)[0]['status'], 'error')
        self.assertEqual(self.store.get(goal_id)['status'], 'active')
        action = self.store.actions(goal_id)[0]
        self.store.resolve(goal_id, action['id'], 'retry')
        self.permission.return_value = False
        self.turn(goal_id, [response(calls=[('write_file', args)]), response('denied')])
        self.assertFalse((self.root / 'first.txt').exists())
        self.assertEqual(self.store.actions(goal_id)[-1]['status'], 'permission_denied')

    def test_checkpoint_is_report_not_observation_or_completion(self):
        goal_id = self.start()
        self.turn(goal_id, [response(calls=[('checkpoint_goal', {'progress':'Everything done','next_action':''})]), response('All done')])
        self.assertEqual(self.store.get(goal_id)['status'], 'active')
        self.assertIn('Unverified model progress: Everything done', self.store.summary(goal_id))
        self.assertFalse((self.root / 'first.txt').exists())

    def test_store_failure_before_dispatch_prevents_effects(self):
        goal_id = self.start()
        with patch.object(self.store, 'begin_action', side_effect=RuntimeError('disk full')):
            with self.assertRaises(RuntimeError):
                self.turn(goal_id, [response(calls=[('write_file', {'path':str(self.root / 'first.txt'), 'content':'first'})])])
        self.assertFalse((self.root / 'first.txt').exists())

    def test_summary_preserves_old_uncertain_action_and_remains_bounded(self):
        goal_id = self.start()
        pending = self.store.begin_action(goal_id, 'write_file', {}, 'call')
        for index in range(20):
            action = self.store.begin_action(goal_id, 'read_file', {'path':str(index)}, 'call')
            self.store.finish_action(action, 'ok', 'result', 'x' * 5000)
        self.store.checkpoint(goal_id, 'p' * 9000, 'n' * 9000)
        summary = self.store.summary(goal_id)
        self.assertIn(pending, summary)
        self.assertIn('n' * 600, summary)
        self.assertLess(len(summary), 6500)

    def test_existing_tool_failure_formats(self):
        for result, tool in [('TERMINAL_ERROR: timed out', 'run_terminal'),
                             ('OPEN_APP_ERROR: missing', 'open_app'),
                             ('Exit code: 1\nSTDOUT: no', 'run_terminal'),
                             ('Path does not exist: nowhere', 'list_directory')]:
            with self.subTest(result=result):
                self.assertEqual(llm._tool_result_status(result, tool), 'error')
        self.assertEqual(llm._tool_result_status('Exit code: 0', 'run_terminal'), 'ok')

    def test_restart_hydrates_terminal_operation_detail_from_old_episode(self):
        goal_id = self.start()
        command = "Set-Content -LiteralPath 'unique-success.txt' -Value 'success'"
        pending_command = "Set-Content -LiteralPath 'unique-interrupted.txt' -Value 'pending'"
        with patch.dict(llm.TOOL_MAP, {'run_terminal':lambda **kw: 'Exit code: 0'}):
            self.turn(goal_id, [response(calls=[('run_terminal', {'command':command})]), response('step')])
        with patch.dict(llm.TOOL_MAP, {'run_terminal':lambda **kw: (_ for _ in ()).throw(KeyboardInterrupt())}):
            with self.assertRaises(KeyboardInterrupt):
                self.turn(goal_id, [response(calls=[('run_terminal', {'command':pending_command})])])
        # Later read-only activity must not hide successful or uncertain mutations.
        for index in range(12):
            self.turn(goal_id, [response(calls=[('read_file', {'path':str(self.root / str(index))})]), response('read')])
        experience.end_episode(reason='restart')
        experience.start_episode()
        with patch.object(goals, '_STORE', goals.GoalStore(self.path)):
            _, actor = self.turn(goal_id, [response('Ready to inspect')])
            context = actor.call_args.kwargs['messages'][1]['content']
            summary = goals.goal_summary(goal_id)
        for text in (command, pending_command):
            self.assertIn(text, context)
            self.assertIn(text, summary)
        self.assertLess(len(summary), 16000)

    def test_interrupted_large_write_keeps_target_visible_on_restart(self):
        goal_id = self.start()
        target = str(self.root / 'unique-pending-write-target.txt')
        arguments = {'content':'x' * 4000, 'path':target}
        with patch.dict(llm.TOOL_MAP, {'write_file':lambda **kw: (_ for _ in ()).throw(KeyboardInterrupt())}):
            with self.assertRaises(KeyboardInterrupt):
                self.turn(goal_id, [response(calls=[('write_file', arguments)])])
        experience.end_episode(reason='restart')
        experience.start_episode()
        with patch.object(goals, '_STORE', goals.GoalStore(self.path)):
            with patch('builtins.input', return_value='quit'), patch('builtins.print') as printed:
                main.main()
            startup = '\n'.join(str(arg) for call in printed.call_args_list for arg in call.args)
            _, actor = self.turn(goal_id, [response('Inspecting')])
            context = actor.call_args.kwargs['messages'][1]['content']
            summary = goals.goal_summary(goal_id)
        for display in (startup, context, summary):
            self.assertIn(target, display)
            self.assertNotIn('x' * 1000, display)


if __name__ == '__main__':
    unittest.main()
