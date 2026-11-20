"""Synthetic end-to-end Day 1 failure and authorization regressions."""
from contextlib import ExitStack
import io
from pathlib import Path
import sys
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from isolation_support import install_guards, TestTemporaryDirectory
install_guards()
import experience
import goals
import llm
import main
import memory
import permissions
import reflection
from inference import InferenceError
from task_controller import Assessment


def response(content='', calls=()):
    return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[
        SimpleNamespace(function=SimpleNamespace(name=name, arguments=args)) for name, args in calls]))


def conversation(content):
    return response(f'{llm.CONVERSATION_PREFIX} {content}')


class ReliabilityRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = TestTemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.memories = memory.MemoryStore(self.root / 'memory.db')
        self.events = experience.ExperienceStore(self.root / 'experience.db')
        for module, name, value in (
            (memory, '_STORE', self.memories), (experience, '_STORE', self.events),
            (goals, '_STORE', goals.GoalStore(self.root / 'goals.db')),
            (experience, '_CURRENT_EPISODE_ID', None),
            (reflection, '_casual_turns_since_reflection', 0),
            (llm, 'ENABLE_DYNAMIC_SELF', False), (llm, 'ENABLE_RELATIONSHIP', False),
            (permissions, 'MODE', 'MEMORY_ASSISTANT'), (permissions, 'WORKSPACE', self.root),
        ):
            self.stack.enter_context(patch.object(module, name, value))

    def events_of_type(self, kind):
        return [e for e in self.events.episode_events(experience.get_current_episode_id(), limit=100)
                if e['event_type'] == kind]

    def test_cli_preserves_answer_and_next_turn_after_reflection_exception_or_timeout(self):
        for error in (RuntimeError('private payload must not leak'), TimeoutError('private timeout')):
            with self.subTest(error=type(error).__name__), \
                    patch.object(llm, 'ollama_chat', side_effect=[conversation('FIRST SYNTHETIC ANSWER'), conversation('SECOND SYNTHETIC ANSWER')]), \
                    patch.object(llm, 'maybe_reflect', side_effect=[error, None]), \
                    patch('builtins.input', side_effect=['Explain the synthetic example', 'Continue the synthetic example', 'exit']), \
                    patch('sys.stdout', new_callable=io.StringIO) as output:
                main.main()
            text = output.getvalue()
            self.assertEqual(text.count('FIRST SYNTHETIC ANSWER'), 1)
            self.assertEqual(text.count('SECOND SYNTHETIC ANSWER'), 1)
            self.assertEqual(text.count('[WARNING] Memory updating failed'), 1)
            self.assertNotIn('private', text)
            self.assertNotIn('MEMORY_SAVED', text)
            self.assertEqual(self.memories.recent_records(), [])

    def test_malformed_real_reflection_pipeline_warns_without_writing_and_recovers(self):
        messages = [{'role': 'user', 'content': 'Synthetic user prefers quiet rooms while studying mathematics every afternoon.'}]
        with patch.object(reflection, 'REFLECTION_INTERVAL', 1), \
                patch.object(reflection, 'ollama_chat', return_value=response('not valid reflection JSON')), \
                patch.object(llm, 'ollama_chat', return_value=response('The synthetic answer is complete.')):
            answer = llm.chat(messages)
        self.assertIn('The synthetic answer is complete.', answer)
        self.assertIn('[WARNING] Memory updating failed', answer)
        self.assertNotIn('MEMORY_SAVED', answer)
        self.assertEqual(len(self.events_of_type('assistant_message')), 1)
        self.assertEqual(len(self.events_of_type('memory_update_failed')), 1)
        self.assertEqual(self.memories.recent_records(), [])
        messages.append({'role': 'user', 'content': 'A subsequent synthetic turn'})
        with patch.object(llm, 'ollama_chat', return_value=response('Recovered.')), patch.object(llm, 'maybe_reflect', return_value=None):
            self.assertEqual(llm.chat(messages), 'Recovered.')

    def test_answer_recording_failure_keeps_cli_answer_and_next_turn(self):
        original = llm.record_assistant_message
        attempts = []

        def record(*args, **kwargs):
            attempts.append(args)
            if len(attempts) == 1:
                raise sqlite3.OperationalError('synthetic private disk failure')
            return original(*args, **kwargs)

        with patch.object(llm, 'record_assistant_message', side_effect=record), \
                patch.object(llm, 'ollama_chat', side_effect=[conversation('COMPLETED FIRST'), conversation('COMPLETED NEXT')]), \
                patch.object(llm, 'maybe_reflect', return_value=None) as reflect, \
                patch('builtins.input', side_effect=['Synthetic first', 'Synthetic next', 'exit']), \
                patch('sys.stdout', new_callable=io.StringIO) as output:
            main.main()
        text = output.getvalue()
        self.assertEqual(text.count('COMPLETED FIRST'), 1)
        self.assertEqual(text.count('COMPLETED NEXT'), 1)
        self.assertEqual(text.count('Memory updating failed'), 1)
        self.assertNotIn('private', text)
        reflect.assert_called_once()
        episode = experience.get_latest_episode_id()
        events = self.events.episode_events(episode, limit=100)
        observed = [e['content'] for e in events if e['event_type'] == 'assistant_message']
        self.assertEqual(observed, ['COMPLETED NEXT'])
        failure = next(e for e in events if e['event_type'] == 'memory_update_failed')
        self.assertEqual(failure['metadata']['stage'], 'answer_recording')
        self.assertEqual(self.memories.recent_records(), [])

    def test_bounded_goal_cli_displays_completed_answer_and_warning_once(self):
        goals._STORE.create('Explain an original synthetic example')
        with patch.object(llm, 'ollama_chat', side_effect=[response('BOUNDED SYNTHETIC ANSWER'), conversation('NEXT ORDINARY ANSWER')]), \
                patch.object(llm, 'assess_task', return_value=Assessment('insufficient_evidence', 'Synthetic no evidence.')), \
                patch.object(llm, 'maybe_reflect', side_effect=[RuntimeError('synthetic reflection failure'), None]), \
                patch('builtins.input', side_effect=['/goal run 1', 'A next synthetic turn', 'exit']), \
                patch('sys.stdout', new_callable=io.StringIO) as output:
            main.main()
        text = output.getvalue()
        self.assertEqual(text.count('BOUNDED SYNTHETIC ANSWER'), 1)
        self.assertEqual(text.count('Memory updating failed'), 1)
        self.assertEqual(text.count('NEXT ORDINARY ANSWER'), 1)
        self.assertEqual(goals._STORE.unfinished()['status'], 'active')
        self.assertEqual(self.memories.recent_records(), [])

    def test_failed_reflection_write_is_not_reported_saved(self):
        result = {'final': {'user_memory': {'content': 'Synthetic preferred number is seven.',
                                            'category': 'preference', 'importance': 3,
                                            'needs_confirmation': False}, 'self_memory': None}}
        with patch.object(reflection, 'REQUIRE_REFLECTION_USER_MEMORY_APPROVAL', False), \
                patch.object(reflection, 'REFLECTION_INTERVAL', 1), \
                patch.object(reflection, 'analyze_reflection_debug', return_value=result), \
                patch.object(reflection, 'save_reflection_memory', side_effect=OSError('synthetic disk failure')), \
                patch.object(llm, 'ollama_chat', return_value=response('Completed answer.')), \
                patch('sys.stdout', new_callable=io.StringIO) as output:
            answer = llm.chat([{'role': 'user', 'content': 'Remember this original synthetic preference: the preferred number is seven.'}])
        self.assertIn('Memory updating failed', answer)
        self.assertNotIn('MEMORY_SAVED', answer + output.getvalue())
        self.assertEqual(self.memories.recent_records(), [])
        self.assertEqual(self.events_of_type('memory_formation'), [])

    def test_missing_actor_model_fails_without_success_and_next_turn_recovers(self):
        with patch.object(llm, 'ollama_chat', side_effect=InferenceError('missing_model')) as actor:
            messages = [{'role': 'user', 'content': 'A synthetic question'}]
            answer = llm.chat(messages)
        self.assertIn('Actor failed', answer)
        self.assertEqual(actor.call_count, 1)
        self.assertEqual(len(messages), 1)
        self.assertEqual(self.events_of_type('assistant_message'), [])
        self.assertEqual(self.memories.recent_records(), [])
        with patch.object(llm, 'ollama_chat', return_value=response('Available now.')), patch.object(llm, 'maybe_reflect', return_value=None):
            self.assertEqual(llm.chat(messages), 'Available now.')

    def test_actor_programming_error_and_process_interrupt_are_not_swallowed(self):
        for error in (TypeError('bug'), KeyboardInterrupt()):
            with patch.object(llm, 'ollama_chat', side_effect=error), self.assertRaises(type(error)):
                llm.chat([{'role': 'user', 'content': 'Synthetic request'}])
        with patch.object(llm, 'ollama_chat', return_value=response('Valid answer')), \
                patch.object(llm, 'maybe_reflect', side_effect=KeyboardInterrupt()), self.assertRaises(KeyboardInterrupt):
            llm.chat([{'role': 'user', 'content': 'Synthetic request'}])

    def test_default_dispatch_denies_host_outside_workspace_and_unknown_tools(self):
        cases = [('run_terminal', {'command': 'never executed'}), ('open_app', {'app_name': 'never executed'}),
                 ('read_file', {'path': 'never read'}), ('list_directory', {'path': 'never read'}),
                 ('write_file', {'path': 'never written', 'content': 'synthetic'}),
                 ('read_source', {'root': str(self.root.parent), 'path': 'never read'}),
                 ('unregistered_tool', {})]
        for name, arguments in cases:
            tool = Mock()
            with self.subTest(tool=name), patch.dict(llm.TOOL_MAP, {name: tool} if name != 'unregistered_tool' else {}), \
                    patch.object(llm, 'ollama_chat', side_effect=[response(calls=[(name, arguments)]), response('Denied operation.')]), \
                    patch.object(llm, 'assess_task', return_value=SimpleNamespace(status='invalid', rationale='denied', as_dict=lambda: {})), \
                    patch.object(llm, 'maybe_reflect', return_value=None), patch.object(permissions, 'ask') as ask:
                llm.chat([{'role': 'user', 'content': 'Synthetic denied dispatch'}])
            tool.assert_not_called()
            ask.assert_not_called()
        self.assertFalse(permissions.check_permission('unregistered_tool'))
        self.assertTrue(permissions.check_permission('read_source', arguments={'root': str(self.root)}))
        self.assertFalse(permissions.check_permission('read_source', arguments={'root': '..'}))

    def test_advanced_opt_in_and_authorized_memory_keep_working(self):
        self.assertEqual(permissions.resolve_mode({}), 'MEMORY_ASSISTANT')
        self.assertEqual(permissions.resolve_mode({'APERTURE_TOOL_PROFILE': 'advanced'}), 'AUTONOMOUS')
        with self.assertRaises(ValueError):
            permissions.resolve_mode({'APERTURE_TOOL_PROFILE': 'typo'})
        with patch.object(permissions, 'MODE', 'AUTONOMOUS'):
            self.assertTrue(permissions.check_permission('run_terminal'))
            self.assertFalse(permissions.check_permission('unregistered_tool'))
        terminal = Mock(return_value='Exit code: 0\nSynthetic terminal evidence')
        with patch.object(permissions, 'MODE', permissions.resolve_mode({'APERTURE_TOOL_PROFILE': 'advanced'})), \
                patch.dict(llm.TOOL_MAP, {'run_terminal': terminal}), \
                patch.object(llm, 'ollama_chat', side_effect=[response(calls=[('run_terminal', {'command': 'synthetic command'})]), response('Synthetic advanced operation completed.')]), \
                patch.object(llm, 'assess_task', return_value=Assessment('insufficient_evidence', 'Synthetic assessment.')), \
                patch.object(llm, 'maybe_reflect', return_value=None):
            answer = llm.chat([{'role': 'user', 'content': 'Explicit synthetic advanced request'}])
        self.assertIn('could not verify completion', answer)
        self.assertIn('tool reported success', answer)
        self.assertNotIn('Synthetic advanced operation completed.', answer)
        terminal.assert_called_once_with(command='synthetic command')
        with patch.object(llm, 'ollama_chat', side_effect=[response(calls=[('save_memory', {'content': 'Synthetic favorite shape is a hexagon.'})]), response('Saved synthetic preference.')]), \
                patch.object(llm, 'maybe_reflect', return_value=None):
            answer = llm.chat([{'role': 'user', 'content': 'Remember my synthetic favorite shape is a hexagon.'}])
        self.assertIn('pending review', answer)
        self.assertEqual(self.memories.recent_records(), [])
        proposal, = self.memories.list_pending_user_memories()
        user_event = self.events_of_type('user_message')[-1]
        self.assertEqual(proposal['evidence_event_ids'], [user_event['id']])
        self.assertIn('hexagon', user_event['content'])
        main.handle_memory_command(f"/memory approve {proposal['id']}")
        stored = self.memories.recent_records()
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]['subject'], 'user')
        self.assertIn('hexagon', stored[0]['content'])
        self.assertEqual(stored[0]['evidence_event_ids'], [user_event['id']])
        self.assertNotIn('run_terminal', {t.__name__ for t in llm.get_tools_for_goal('hello')})


if __name__ == '__main__':
    unittest.main()
