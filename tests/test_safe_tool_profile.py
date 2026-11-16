"""Default tool authority is enforced at advertisement and dispatch."""
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from isolation_support import TestTemporaryDirectory, install_guards

install_guards()

import experience  # noqa: E402
import goals  # noqa: E402
import llm  # noqa: E402
import memory  # noqa: E402
import permissions  # noqa: E402
from task_controller import Assessment  # noqa: E402


def response(content='', calls=()):
    return SimpleNamespace(message=SimpleNamespace(
        content=content,
        tool_calls=[SimpleNamespace(function=SimpleNamespace(name=name, arguments=args))
                    for name, args in calls],
    ))


class SafeToolProfileTests(unittest.TestCase):
    def setUp(self):
        directory = TestTemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = root
        self.memories = memory.MemoryStore(root / 'memory.db')
        self.events = experience.ExperienceStore(root / 'experience.db')
        for module, name, value in (
            (memory, '_STORE', self.memories),
            (experience, '_STORE', self.events),
            (experience, '_CURRENT_EPISODE_ID', None),
            (goals, '_STORE', goals.GoalStore(root / 'goals.db')),
            (permissions, 'MODE', 'MEMORY_ASSISTANT'),
            (permissions, 'WORKSPACE', root),
            (llm, 'ENABLE_DYNAMIC_SELF', False),
            (llm, 'ENABLE_RELATIONSHIP', False),
        ):
            stack.enter_context(patch.object(module, name, value))
        stack.enter_context(patch.object(
            llm, 'inject_runtime_context', side_effect=lambda messages, **kw: list(messages)))
        stack.enter_context(patch.object(llm, 'maybe_reflect', return_value=None))
        stack.enter_context(patch.object(
            llm, 'assess_task', return_value=Assessment('insufficient_evidence', 'Synthetic test.')))

    def test_default_schema_excludes_general_host_tools(self):
        self.assertEqual(permissions.resolve_mode({}), 'MEMORY_ASSISTANT')
        names = {tool.__name__ for tool in llm.get_tools_for_goal('Inspect this project')}
        self.assertTrue({'discover_workspace', 'read_source', 'search_memory'} <= names)
        self.assertFalse(names & permissions.HOST_TOOLS)

    def test_forged_host_call_is_denied_before_prompt_or_effect(self):
        for name, args in (
            ('run_terminal', {'command': 'synthetic command'}),
            ('open_app', {'app_name': 'synthetic application'}),
            ('read_file', {'path': 'synthetic file'}),
            ('write_file', {'path': 'synthetic file', 'content': 'synthetic'}),
            ('list_directory', {'path': 'synthetic directory'}),
        ):
            with self.subTest(tool=name):
                effect = Mock(return_value='synthetic effect')
                with patch.dict(llm.TOOL_MAP, {name: effect}), \
                        patch.object(llm, 'ollama_chat', side_effect=[
                            response(calls=[(name, args)]), response('Operation was denied.')]), \
                        patch.object(permissions, 'ask') as ask:
                    answer = llm.chat([{'role': 'user', 'content': 'Synthetic denied request'}])
                self.assertIn('denied', answer.lower())
                effect.assert_not_called()
                ask.assert_not_called()

    def test_scoped_tools_require_structured_workspace_root(self):
        inside = str(self.root)
        outside = str(self.root.parent)
        for arguments in (None, {}, {'root': 42}, {'root': '..'},
                          {'root': outside}):
            with self.subTest(arguments=arguments):
                self.assertFalse(permissions.check_permission(
                    'read_source', arguments=arguments))
        self.assertTrue(permissions.check_permission(
            'read_source', arguments={'root': inside, 'path': 'sample.txt'}))
        with patch.object(permissions, 'ask', return_value=False) as ask:
            self.assertFalse(permissions.check_permission(
                'create_text', arguments={'root': inside, 'path': 'sample.txt'}))
        ask.assert_called_once()

    def test_advanced_requires_explicit_profile_and_unknown_stays_denied(self):
        self.assertEqual(permissions.resolve_mode({'APERTURE_TOOL_PROFILE': 'advanced'}),
                         'AUTONOMOUS')
        with self.assertRaises(ValueError):
            permissions.resolve_mode({'APERTURE_TOOL_PROFILE': 'other'})
        self.assertFalse(permissions.check_permission('unknown_tool'))
        with patch.object(permissions, 'MODE', 'AUTONOMOUS'):
            self.assertIn('run_terminal', {
                tool.__name__ for tool in llm.get_tools_for_goal('Synthetic advanced request')})
            self.assertTrue(permissions.check_permission('run_terminal'))
            self.assertFalse(permissions.check_permission('unknown_tool'))

    def test_default_forget_requires_approval_before_memory_change(self):
        record = self.memories.remember('Synthetic fact to retain.', subject='user')
        with patch.object(permissions, 'ask', return_value=False) as ask, \
                patch.object(llm, 'ollama_chat', side_effect=[
                    response(calls=[('forget_memory', {'memory_id': record['id']})]),
                    response('The removal was denied.')]):
            answer = llm.chat([{'role': 'user', 'content': 'What do you remember?'}])
        self.assertIn('denied', answer.lower())
        ask.assert_called_once_with('forget_memory', record['id'])
        self.assertTrue(self.memories.get_by_id(record['id'])['active'])
        with patch.object(permissions, 'ask', return_value=True) as ask, \
                patch.object(llm, 'ollama_chat', side_effect=[
                    response(calls=[('forget_memory', {'memory_id': record['id']})]),
                    response('The approved removal completed.')]):
            llm.chat([{'role': 'user', 'content': 'Forget that synthetic fact.'}])
        ask.assert_called_once_with('forget_memory', record['id'])
        self.assertFalse(self.memories.get_by_id(record['id'])['active'])

    def test_authorized_memory_write_keeps_observed_user_provenance(self):
        message = 'Remember that my synthetic preferred shape is a hexagon.'
        with patch.object(llm, 'ollama_chat', side_effect=[
            response(calls=[('save_memory', {'content':
                         'The user prefers hexagons.', 'category': 'preference'})]),
            response('The synthetic preference is pending review.')]):
            answer = llm.chat([{'role': 'user', 'content': message}])
        self.assertIn('pending review', answer)
        stored = self.memories.list_pending_user_memories()
        self.assertEqual(len(stored), 1)
        self.assertEqual(self.memories.recent_records(), [])
        source = experience.get_events_by_ids(stored[0]['evidence_event_ids'])
        self.assertEqual(len(source), 1)
        self.assertEqual(source[0]['event_type'], 'user_message')
        self.assertEqual(source[0]['actor'], 'arda')
        self.assertEqual(source[0]['content'], message)


if __name__ == '__main__':
    unittest.main()
