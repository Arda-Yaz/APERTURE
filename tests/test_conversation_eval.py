import json
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from isolation_support import TestTemporaryDirectory, install_guards

install_guards()
spec = importlib.util.spec_from_file_location('conversation_eval', Path(__file__).resolve().parents[1] / 'tools/conversation_eval.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class ConversationEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = patch.object(evaluation.tempfile, 'TemporaryDirectory', TestTemporaryDirectory)
        temporary.start()
        self.addCleanup(temporary.stop)

    def test_budget_and_production_output_rejected_before_imports_or_inference(self):
        client = Mock()
        with patch.object(evaluation, 'synthetic_runtime') as runtime:
            with self.assertRaises(ValueError):
                evaluation.evaluate('unused.json', live=True, max_calls=1, client=client)
            with self.assertRaises(ValueError):
                evaluation.evaluate(evaluation.ROOT / 'data' / 'eval.json')
            runtime.assert_not_called()
        client.chat.assert_not_called()

    def test_actual_context_and_ablation_keep_history_and_tools(self):
        with TestTemporaryDirectory() as directory, evaluation.synthetic_runtime(directory) as (llm, bridge):
            item = evaluation.CASES[0]
            normal = evaluation.build_request(llm, bridge, item, 'default', 17, 300)
            legacy = evaluation.build_request(llm, bridge, item, 'legacy_session_bridge', 17, 300)
            compact = evaluation.build_request(llm, bridge, item, 'compact_persona', 17, 300)
            no_bridge = evaluation.build_request(llm, bridge, item, 'no_session_bridge', 17, 300)
            no_memory = evaluation.build_request(llm, bridge, item, 'no_memory', 17, 300)
            minimal = evaluation.build_request(llm, bridge, item, 'minimal', 17, 300)
            no_persona = evaluation.build_request(llm, bridge, item, 'no_persona', 17, 300)
            candidate = evaluation.build_request(llm, bridge, item, 'candidate_foundation', 17, 300)
            history = evaluation.build_request(llm, bridge, item, 'candidate_history', 17, 300)
            combined = evaluation.build_request(llm, bridge, item, 'candidate_combined', 17, 300)
            no_penalty = evaluation.build_request(llm, bridge, item, 'no_presence_penalty', 17, 300)
            no_tools = evaluation.build_request(llm, bridge, item, 'no_tools', 17, 300)
            no_thinking = evaluation.build_request(llm, bridge, item, 'no_thinking', 17, 300)
            override = evaluation.build_request(llm, bridge, item, 'minimal', 17, 300, model='hermes3:8b', no_think=True)
            self.assertIn('<RECENT_SESSION_CONTEXT>', legacy['messages'][0]['content'])
            self.assertIn('<RECENT_SESSION_CONTEXT>', compact['messages'][0]['content'])
            self.assertIn(evaluation.COMPACT_PERSONA, compact['messages'][0]['content'])
            self.assertNotIn('Your pretrained conversational tendencies', compact['messages'][0]['content'])
            self.assertEqual(compact['messages'][1:], legacy['messages'][1:])
            self.assertEqual(compact['tools'], legacy['tools'])
            self.assertEqual(compact['options'], legacy['options'])
            self.assertIn('<RECENT_SESSION_CONTEXT>', normal['messages'][0]['content'])
            self.assertEqual(normal, legacy)
            self.assertIn('<PERSONA_CONTEXT>', normal['messages'][0]['content'])
            self.assertNotIn('<RECENT_SESSION_CONTEXT>', no_bridge['messages'][0]['content'])
            self.assertIn('<ACTOR_MEMORY>', normal['messages'][0]['content'])
            self.assertNotIn('<ACTOR_MEMORY>', no_memory['messages'][0]['content'])
            self.assertEqual(minimal['messages'][0]['content'], evaluation.MINIMAL)
            self.assertNotIn('<PERSONA_CONTEXT>', no_persona['messages'][0]['content'])
            self.assertIn('<RECENT_SESSION_CONTEXT>', no_persona['messages'][0]['content'])
            self.assertIn(evaluation.CANDIDATE_FOUNDATION, candidate['messages'][0]['content'])
            self.assertIn('<PERSONA_CONTEXT>', candidate['messages'][0]['content'])
            self.assertEqual(candidate['messages'][1:], normal['messages'][1:])
            self.assertNotIn('<RECENT_SESSION_CONTEXT>', history['messages'][0]['content'])
            self.assertEqual(history['messages'][1:3], evaluation.PREVIOUS_HISTORY)
            self.assertEqual(history['messages'][3:], normal['messages'][1:])
            self.assertIn('Prior requests are not current authorization', history['messages'][0]['content'])
            self.assertEqual(history['tools'], normal['tools'])
            self.assertEqual(history['options'], normal['options'])
            self.assertIn(evaluation.CANDIDATE_FOUNDATION, combined['messages'][0]['content'])
            self.assertEqual(combined['messages'][1:], history['messages'][1:])
            self.assertEqual(no_penalty['messages'], normal['messages'])
            self.assertEqual(no_penalty['options'], {**normal['options'], 'presence_penalty': 0.0})
            self.assertEqual(no_tools, {**normal, 'tools': []})
            self.assertEqual(no_thinking, {**normal, 'think': False})
            self.assertEqual(override, {**minimal, 'model': 'hermes3:8b', 'think': False})
            self.assertEqual(normal['messages'][1:], minimal['messages'][1:])
            self.assertEqual(normal['tools'], no_memory['tools'])
            names = {schema['function']['name'] for schema in normal['tools']}
            self.assertIn('search_memory', names)
            self.assertIn('read_source', names)
            self.assertFalse(names & {'run_terminal', 'open_app', 'read_file', 'list_directory', 'write_file'})
            self.assertEqual(minimal['tools'], [])
            self.assertEqual(normal['options']['seed'], 17)
            self.assertEqual(normal['options']['num_predict'], 300)

    def test_live_response_only_records_tool_call_never_dispatches(self):
        client = Mock()
        client.list.return_value.model_dump.return_value = {'models': [{'model': 'synthetic'}]}
        client.show.return_value.model_dump.return_value = {'details': {'family': 'test'}}
        client.ps.return_value.model_dump.return_value = {'models': [{'context_length': 4096}]}
        response = {'message': {'content': '', 'tool_calls': [{'function': {'name': 'write_file', 'arguments': {'path': 'never'}}}]},
                    'done_reason': 'length', 'eval_count': 5}
        client.chat.return_value.model_dump.return_value = response
        with TestTemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            report = evaluation.evaluate(output, live=True, modes=['default'], case_ids=['correction'],
                                         seeds=[12], num_predict=8, max_calls=1, client=client)
            record = report['results'][0]
            self.assertTrue(record['tool_calls_suppressed'])
            self.assertTrue(record['truncated'])
            self.assertTrue(record['empty_answer'])
            self.assertEqual(record['response'], response)
            self.assertEqual(json.loads(output.read_text(encoding='utf-8')), report)
            self.assertEqual(client.chat.call_count, 1)
            self.assertEqual(report['loaded_models_after']['models'][0]['context_length'], 4096)
            self.assertIn('scenario', record['criterion'])

    def test_network_errors_preserved_and_dry_run_never_contacts_client(self):
        client = Mock()
        client.list.return_value.model_dump.return_value = {}
        client.show.return_value.model_dump.return_value = {}
        client.ps.return_value.model_dump.return_value = {}
        client.chat.side_effect = TimeoutError('bounded timeout')
        with TestTemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            report = evaluation.evaluate(output, live=True, modes=['minimal'], case_ids=['honest_cause'],
                                         max_calls=1, client=client)
            self.assertIn('TimeoutError', report['results'][0]['error'])
            client.reset_mock()
            dry = evaluation.evaluate(output, modes=['minimal'], case_ids=['honest_cause'], max_calls=1, client=client)
            client.chat.assert_not_called()
            client.list.assert_not_called()
            self.assertNotIn('response', dry['results'][0])

    def test_sqlite_guard_rejects_production_even_when_app_catches_exception(self):
        import sqlite3
        with TestTemporaryDirectory() as directory, evaluation.synthetic_runtime(directory):
            with self.assertRaisesRegex(AssertionError, 'outside temporary'):
                sqlite3.connect(evaluation.ROOT / 'data' / 'aperture_memory.db')
            with self.assertRaisesRegex(AssertionError, 'outside temporary'):
                sqlite3.connect('file:memory?mode=memory', uri=True)


if __name__ == '__main__':
    unittest.main()
