from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from isolation_support import TestTemporaryDirectory, install_guards

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import turn_latency_eval


class TurnLatencyTests(unittest.TestCase):
    def test_budget_preserves_smaller_request_and_caps_larger_one(self):
        self.assertEqual(turn_latency_eval.budget_options({'num_predict': 16, 'seed': 42}, 2048),
                         {'num_predict': 16, 'seed': 42})
        self.assertEqual(turn_latency_eval.budget_options({'num_predict': 4096}, 2048), {'num_predict': 2048})
        self.assertEqual(turn_latency_eval.budget_options(None, 2048), {'num_predict': 2048})

    def test_runtime_budget_uses_same_effective_options_as_inference(self):
        install_guards()
        import inference

        client = MagicMock()
        client.__enter__.return_value.chat.return_value = SimpleNamespace(message=SimpleNamespace(content='answer'))
        with patch.dict('os.environ', {'APERTURE_INFERENCE_NUM_PREDICT_RETRIEVAL': '256'}), \
                patch.object(inference.ollama, 'Client', return_value=client):
            for options in (None, {'num_predict': 512, 'seed': 42}, {'num_predict': 16}):
                inference.chat(role='retrieval', options=options)
                actual = client.__enter__.return_value.chat.call_args.kwargs['options']
                self.assertEqual(actual, turn_latency_eval.budget_options(options, 256))

    def test_offline_probe_exercises_each_scheduled_phase(self):
        with TestTemporaryDirectory() as directory:
            report = turn_latency_eval.evaluate(Path(directory) / 'turns.json', live=False, samples=1)

        self.assertTrue(report['complete'])
        self.assertEqual(report['live_calls'], 0)
        self.assertEqual(report['guard_violations'], [])
        self.assertEqual(len(report['results']), 6)
        for group, expected in {
            '0_ordinary': (1, 0, 0, 0), '149_ordinary': (1, 1, 0, 0),
            '0_reflection_due': (1, 0, 2, 0), '149_reflection_due': (1, 2, 2, 0),
            '0_file_conversation': (1, 0, 0, 1), '149_file_conversation': (1, 1, 0, 1),
        }.items():
            with self.subTest(group=group):
                results = report['groups'][group]
                self.assertEqual(results['failed_turns'], 0)
                self.assertEqual(results['unverified_replies'], 0)
                for role, calls in zip(turn_latency_eval.TOKEN_LIMITS, expected):
                    bucket = results['inference'][role]
                    self.assertEqual(bucket['calls'], calls)
                    self.assertEqual(bucket['seconds_per_active_turn'] is None, not calls)

    def test_unsolicited_actor_tool_call_cannot_reach_dispatch(self):
        install_guards()
        import llm
        import task_controller

        tool = Mock()
        permission = Mock(return_value=True)
        call = SimpleNamespace(function=SimpleNamespace(name='read_source', arguments={'path': 'outside.txt'}))
        response = SimpleNamespace(message=SimpleNamespace(content='', tool_calls=[call]))
        with TestTemporaryDirectory() as directory, turn_latency_eval._isolated_stores(Path(directory)) as (
            _, experience, reflection, _, _,
        ), patch.dict(llm.TOOL_MAP, {'read_source': tool}), patch.object(llm, 'check_permission', permission):
            row = turn_latency_eval.measure_turn(llm, task_controller, reflection, experience,
                                                'ordinary', 'Hello.', 0, lambda role, request: response)

        self.assertEqual(row['error'], 'ActorFailed')
        self.assertTrue(row['calls'][0]['unexpected_tool_call'])
        permission.assert_not_called()
        tool.assert_not_called()
