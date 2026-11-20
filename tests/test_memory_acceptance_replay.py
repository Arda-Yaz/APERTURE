"""Offline replay reproduces the recorded approval counts in temporary SQLite stores."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import httpx
import ollama

from isolation_support import TestTemporaryDirectory, install_guards

install_guards()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import memory_acceptance_replay as replay


class MemoryAcceptanceReplayTests(unittest.TestCase):
    def test_frozen_inputs_reject_changed_responses_or_review(self):
        for name in replay.FROZEN_INPUTS:
            with self.subTest(name=name), TestTemporaryDirectory() as directory:
                original = replay.FIXTURES / name
                changed = Path(directory) / name
                changed.write_bytes(original.read_bytes() + b' ')
                with patch.object(replay, 'FIXTURES', Path(directory)):
                    with self.assertRaisesRegex(ValueError, 'Frozen replay input hash changed'):
                        replay._frozen_json(name)

    def test_two_policies_reproduce_archived_counts_without_inference(self):
        with TestTemporaryDirectory() as directory:
            output = Path(directory)
            summary = replay.replay(output, make_temp=TestTemporaryDirectory)
            self.assertEqual(summary['ollama_calls'], 0)
            for policy, matches, wrong, burden, newly_active in (
                    ('automatic', 16, 2, (1, 0, 0, 0, 1), 10),
                    ('queued', 18, 0, (11, 10, 8, 2, 1), 8)):
                report = summary['policies'][policy]
                self.assertEqual(report['replayed_responses'], 40)
                self.assertEqual(report['agent_fidelity'], {'faithful': 9, 'reviewed': 11})
                metrics = report['after_decision']
                self.assertEqual((metrics['cases'], metrics['decision_matches']), (20, matches))
                self.assertEqual((metrics['false_active_writes'],
                                  metrics['false_active_write_eligible_cases']), (wrong, 11))
                self.assertEqual((metrics['false_active_supersessions'],
                                  metrics['false_active_supersession_eligible_cases']), (0, 20))
                self.assertEqual((metrics['correct_seeds_preserved'],
                                  metrics['correct_seed_preservation_eligible_cases']), (2, 2))
                self.assertEqual(tuple(metrics['review_burden'][field] for field in (
                    'cases_with_proposals', 'commands_attempted', 'approved', 'rejected',
                    'still_pending')), burden)
                fidelity = metrics['new_active_case_fidelity']
                self.assertEqual((fidelity['faithful'], fidelity['reviewed']), (8, newly_active))
                raw_path = output / f'{policy}-raw.json'
                raw = json.loads(raw_path.read_text(encoding='utf-8'))
                self.assertFalse(raw['live'])
                self.assertEqual(raw['network_model_calls_attempted'], 0)
                self.assertEqual(raw['frozen_input_sha256'], replay.FROZEN_INPUTS)
                self.assertEqual(report['raw_report_sha256'], replay.evaluator._digest(raw_path))

            packet = json.loads((output / 'human-review-packet.json').read_text(encoding='utf-8'))
            self.assertEqual(packet['reviewer'], 'human')
            self.assertEqual(packet['status'], 'awaiting_independent_review')
            self.assertEqual(packet['report_sha256'], summary['policies']['queued']['raw_report_sha256'])
            self.assertEqual(packet['coverage'], {
                'decisions': 0, 'fidelity': 0, 'duration': 0, 'cases_in_packet': 5,
            })
            self.assertEqual(len(packet['cases']), 5)
            for row in packet['cases'].values():
                self.assertTrue(row['dialogue'])
                self.assertEqual(len(row['proposals']), 1)
                self.assertEqual(row['human_response'], {
                    'decision': None, 'verdict': None, 'reason': None, 'review_seconds': None,
                })
            temporal = packet['cases']['temporal_holdout_tea_to_coffee']
            self.assertIsNotNone(temporal['prior_user_observation'])
            self.assertEqual(temporal['prior_active_memory']['content'],
                             temporal['proposals'][0]['target_content'])

            packet_path = output / 'human-review-packet.json'
            queued_path = output / 'queued-raw.json'
            blank_score = replay.score_human_review(packet_path, queued_path, output / 'blank.json')
            self.assertEqual(blank_score['coverage'], packet['coverage'])
            self.assertEqual(blank_score['status'], 'awaiting_independent_review')
            self.assertIsNone(blank_score['fidelity']['rate'])
            self.assertIsNone(blank_score['median_review_seconds'])

            # Simulated form input exercises intake only.
            returned = deepcopy(packet)
            first, second, third, _, _ = returned['cases'].values()
            first['human_response'].update(
                decision='approve', verdict='faithful', reason='Test form input.', review_seconds=10)
            second['human_response'].update(verdict='unfaithful', reason='Test form input.')
            third['human_response'].update(decision='defer', review_seconds=14)
            returned_path = output / 'returned.json'
            returned_path.write_text(json.dumps(returned), encoding='utf-8')
            score_path = output / 'human-scored.json'
            score = replay.score_human_review(returned_path, queued_path, score_path)
            self.assertEqual(score['coverage'], {
                'decisions': 2, 'fidelity': 2, 'duration': 2, 'cases_in_packet': 5,
            })
            self.assertEqual(score['decision_counts'], {'approve': 1, 'reject': 0, 'defer': 1})
            self.assertEqual(score['fidelity'], {'faithful': 1, 'unfaithful': 1, 'rate': 0.5})
            self.assertEqual(score['median_review_seconds'], 12)
            self.assertEqual(score['queued_report_sha256'], packet['report_sha256'])
            self.assertEqual(score['submitted_packet_sha256'], replay.evaluator._digest(returned_path))
            self.assertEqual(json.loads(score_path.read_text(encoding='utf-8')), score)

            with self.assertRaisesRegex(ValueError, 'unscored queued replay report'):
                replay.score_human_review(
                    returned_path, output / 'automatic-raw.json', output / 'bad.json')
            with self.assertRaisesRegex(ValueError, 'separate from both inputs'):
                replay.score_human_review(returned_path, queued_path, returned_path, overwrite=True)

            def assert_rejects(change, message):
                invalid = deepcopy(returned)
                change(invalid, next(iter(invalid['cases'].values())))
                returned_path.write_text(json.dumps(invalid), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, message):
                    replay.score_human_review(returned_path, queued_path, output / 'bad.json')

            assert_rejects(lambda item, _: item.update(report_sha256='unrelated'), 'complete packet')
            assert_rejects(lambda item, _: item['cases'].popitem(), 'complete packet')
            assert_rejects(lambda _, case: case.update(dialogue=[]), 'evidence changed')
            assert_rejects(lambda _, case: case['human_response'].update(decision='skip'), 'decision')
            assert_rejects(lambda _, case: case['human_response'].update(verdict='maybe'), 'verdict')
            assert_rejects(lambda _, case: case['human_response'].update(reason=None), 'needs a reason')
            for value in (True, -1, float('inf'), float('nan')):
                assert_rejects(lambda _, case: case['human_response'].update(review_seconds=value),
                               'review time')

    def test_replay_api_refuses_production_output(self):
        with self.assertRaisesRegex(ValueError, 'production data'):
            replay.replay(ROOT / 'data' / 'memory-approval-replay')

    def test_replay_blocks_sdk_requests_before_transport_even_when_case_catches_them(self):
        with TestTemporaryDirectory() as directory:
            output = Path(directory)
            probed = False

            def probe(*args):
                nonlocal probed
                if not probed:
                    probed = True
                    with ollama.Client(host='http://127.0.0.1:1', trust_env=False) as sdk:
                        with self.assertRaises(replay.OfflineReplayNetworkError):
                            sdk.list()
                return {}

            with patch.object(replay, '_run_recorded_case', side_effect=probe), \
                    patch.object(httpx.HTTPTransport, 'handle_request',
                                 side_effect=AssertionError('Unexpected HTTP transport')) as transport:
                with self.assertRaisesRegex(replay.OfflineReplayNetworkError, 'Network requests are disabled'):
                    replay.replay(output, make_temp=TestTemporaryDirectory)
            transport.assert_not_called()
            self.assertEqual(list(output.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
