import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import isolation_support
from isolation_support import TestTemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import retrieval_eval
import retrieval_fallback_eval


class RetrievalEvaluationTests(unittest.TestCase):
    def test_offline_report_measures_each_production_path(self):
        retrieval_eval.SCRATCH.mkdir(exist_ok=True)
        with TestTemporaryDirectory(dir=retrieval_eval.SCRATCH) as directory:
            output = Path(directory) / 'report.json'
            report = retrieval_eval.evaluate(output)
            self.assertEqual(report, json.loads(output.read_text(encoding='utf-8')))
            with self.assertRaises(FileExistsError):
                retrieval_eval.evaluate(output, group='historical')
            self.assertEqual(report, json.loads(output.read_text(encoding='utf-8')))

        self.assertEqual(report['model_calls'], 0)
        self.assertEqual(report['history_older_per_owner'], 2)
        self.assertIsNone(report['model'])
        self.assertEqual(report['actor_selector_mode'], 'gold_id_mock')
        self.assertEqual(report['guard_violations'], [])
        self.assertEqual(report['source_sha256']['app/memory.py'],
                         hashlib.sha256((ROOT / 'app/memory.py').read_bytes()).hexdigest())
        self.assertEqual(report['fixture_sha256'], {
            'evaluations/retrieval/baseline.json':
                '904fba12cece60c9a3c1b42b77e68d66ce8a66d33ed551a576bd34011fab91c3',
            'evaluations/retrieval/fresh.json':
                '1621da914d23ca99688a7df2d5c28a01d43f8a6c58d1302f4840e67489586786',
            'evaluations/retrieval/independent.json':
                'e9f13a006594c288fc1d65978751cdcbbc8e91c3e0ddc05e5d850cbddfa65b05',
            'evaluations/retrieval/pressure.json':
                '62c1306a37421705095067fa9f7336e2e19a43b2ed647ef902d62a319d16ef01',
        })
        self.assertEqual(report['corpus_by_group'], {
            'historical': 'evaluations/retrieval/baseline.json',
            'fresh': 'evaluations/retrieval/baseline.json',
            'independent': 'evaluations/retrieval/independent.json',
            'pressure': 'evaluations/retrieval/pressure.json',
        })
        for split, corpus in report['corpus_by_group'].items():
            fixture = json.loads((ROOT / corpus).read_text(encoding='utf-8'))
            record_ids = {item['id'] for item in fixture['records']}
            for path in report['groups'][split].values():
                cost = path['metrics']['selector_cost']
                self.assertLessEqual(cost['max_candidates'], 96)
                self.assertLessEqual(cost['max_candidate_bytes'], 20_000)
                for row in path['results']:
                    self.assertLessEqual(set(row['returned_ids']), record_ids)
                    self.assertLessEqual(set(row['candidate_ids'] or []), record_ids)
                    self.assertLessEqual(set(row['gold_ids']), record_ids)

        paths = report['groups']['fresh']
        self.assertEqual(set(paths), {'lexical_tool', 'actor_fallback', 'actor_selector'})
        for item in paths.values():
            metrics = item['metrics']
            self.assertEqual(metrics['completed_measurements'], 10)
            self.assertEqual(metrics['recall_at_5']['denominator'], 6)
            self.assertEqual(metrics['no_answer_false_positive_rate']['denominator'], 4)
            self.assertEqual(metrics['inactive_returns'], 0)
            self.assertEqual(metrics['ownership_tag_violations'], 0)
            self.assertEqual(metrics['delivered_results'], {
                name: metrics[name] for name in ('recall_at_5', 'mrr_at_5', 'no_answer_false_positive_rate')
            })
        self.assertIsNone(paths['lexical_tool']['results'][0]['candidate_ids'])
        self.assertTrue(paths['actor_fallback']['results'][0]['candidate_ids'])
        for row in paths['actor_selector']['results']:
            self.assertEqual(row['returned_ids'],
                             [key for key in row['gold_ids'] if key in row['candidate_ids']])
        independent = report['groups']['independent']
        for path in independent.values():
            self.assertEqual(path['metrics']['completed_measurements'], 8)
            self.assertEqual(path['metrics']['recall_at_5']['denominator'], 4)
            self.assertEqual(path['metrics']['no_answer_false_positive_rate']['denominator'], 4)
        pressure = report['groups']['pressure']['actor_selector']['metrics']
        self.assertEqual(pressure['old_zero_overlap_coverage'], retrieval_eval.rate(1, 2))
        self.assertEqual(pressure['strong_current_recall_at_5'], retrieval_eval.rate(1, 1))

    def test_independent_fixture_has_crowded_old_records_and_strong_overlap_negatives(self):
        fixture = json.loads((retrieval_eval.FIXTURES / 'independent.json').read_text(encoding='utf-8'))
        records = fixture['records']

        def tokens(text):
            return set(re.findall(r'\w+', text.casefold()))

        older = [case for case in fixture['cases'] if case['kind'] == 'old_zero_overlap']
        self.assertEqual(len(older), 2)
        for case in older:
            index = next(i for i, record in enumerate(records) if record['id'] in case['gold_ids'])
            gold = records[index]
            self.assertFalse(tokens(case['query']) & tokens(gold['content']))
            newer_same_owner = [record for record in records[index + 1:] if record['subject'] == gold['subject']]
            self.assertGreater(len(newer_same_owner), 96)

        negatives = [case for case in fixture['cases'] if case['kind'] == 'strong_overlap_no_answer']
        self.assertEqual(len(negatives), 4)
        for case in negatives:
            self.assertEqual(case['gold_ids'], [])
            overlaps = [len(tokens(case['query']) & tokens(record['content'])) for record in records]
            self.assertGreaterEqual(max(overlaps), 5)

    def test_evaluation_can_replay_the_zero_reserve_baseline(self):
        retrieval_eval.SCRATCH.mkdir(exist_ok=True)
        with TestTemporaryDirectory(dir=retrieval_eval.SCRATCH) as directory:
            report = retrieval_eval.evaluate(Path(directory) / 'reserve.json', group='pressure',
                                             older_per_owner=0)
        paths = report['groups']['pressure']
        metrics = paths['actor_selector']['metrics']
        self.assertEqual(report['history_older_per_owner'], 0)
        self.assertEqual(metrics['old_zero_overlap_coverage'], retrieval_eval.rate(0, 2))
        self.assertEqual(metrics['strong_current_recall_at_5'], retrieval_eval.rate(1, 1))
        self.assertEqual(metrics['selector_cost']['max_candidates'], 96)
        self.assertLessEqual(metrics['selector_cost']['max_candidate_bytes'], 20_000)
        self.assertEqual(report['model_calls'], 0)
        self.assertEqual(report['guard_violations'], [])

    def test_fallback_gates_report_suppressed_negatives_and_lost_gold_separately(self):
        rows = [
            dict(id='sparse_correct', returned_ids=['correct'], gold_ids=['correct'],
                 shared_tokens=1, query_fraction=0.3),
            dict(id='sparse_wrong', returned_ids=['wrong'], gold_ids=[],
                 shared_tokens=1, query_fraction=0.3),
            dict(id='strong_wrong', returned_ids=['strong'], gold_ids=[],
                 shared_tokens=4, query_fraction=0.6),
        ]
        comparison = retrieval_fallback_eval.compare(rows)
        self.assertEqual(comparison['baseline']['recall_at_5'], retrieval_eval.rate(1, 1))
        self.assertEqual(comparison['two_shared_tokens']['lost_correct_ids_at_5'], retrieval_eval.rate(1, 1))
        self.assertEqual(comparison['two_shared_tokens']['suppressed_false_positives'], retrieval_eval.rate(1, 2))
        self.assertEqual(comparison['two_shared_tokens']['no_answer_false_positive_rate'], retrieval_eval.rate(1, 2))

    def test_local_digest_requires_the_requested_model_tag(self):
        models = [SimpleNamespace(model='fixture:latest', digest='one'),
                  SimpleNamespace(model='fixture:tested', digest='two')]
        client = SimpleNamespace(list=lambda: SimpleNamespace(models=models))
        self.assertEqual(retrieval_eval.installed_digest(client, 'fixture'), 'one')
        self.assertEqual(retrieval_eval.installed_digest(client, 'fixture:tested'), 'two')
        with self.assertRaisesRegex(ValueError, 'no installed digest'):
            retrieval_eval.installed_digest(client, 'fixture:untested')

    def test_live_report_distinguishes_effective_settings_and_fallback_delivery(self):
        isolation_support.install_guards()
        import ollama

        clients = []

        class LocalClientDouble:
            def __init__(self, **options):
                clients.append(self)
                self.options = options
                self.inherited_key = os.environ.get('OLLAMA_API_KEY')
                self.requests = []
                self.closed = False

            def list(self):
                return SimpleNamespace(models=[SimpleNamespace(model='fixture:local', digest='digest')])

            def chat(self, **request):
                self.requests.append(request)
                if len(self.requests) in (5, 7):
                    raise TimeoutError('synthetic timeout')
                return SimpleNamespace(message=SimpleNamespace(content='{"memory_ids": []}'))

            def close(self):
                self.closed = True

        retrieval_eval.SCRATCH.mkdir(exist_ok=True)
        with TestTemporaryDirectory(dir=retrieval_eval.SCRATCH) as directory, \
                patch.object(ollama, 'Client', LocalClientDouble), \
                patch.dict(os.environ, OLLAMA_API_KEY='fixture-key',
                           APERTURE_MODEL_RETRIEVAL='fixture:runtime',
                           APERTURE_INFERENCE_NUM_PREDICT_RETRIEVAL='64',
                           APERTURE_INFERENCE_TIMEOUT_SECONDS_RETRIEVAL='17',
                           APERTURE_INFERENCE_RETRIES_RETRIEVAL='2'), \
                patch.object(isolation_support, '_INSTALLED', False), \
                patch.object(isolation_support, 'install_guards'):
            # Stand in for the standalone CLI's initialization. The storage
            # guards stay installed and the client above cannot use a network.
            report = retrieval_eval.evaluate(Path(directory) / 'report.json',
                                             live=True, model='fixture:local', group='fresh')
            self.assertEqual(os.environ['OLLAMA_API_KEY'], 'fixture-key')

        client, = clients
        self.assertTrue(client.closed)
        self.assertIsNone(client.inherited_key)
        self.assertEqual(client.options['host'], 'http://127.0.0.1:11434')
        self.assertFalse(client.options['trust_env'])
        self.assertFalse(client.options['follow_redirects'])
        settings = report['model']
        self.assertEqual(settings['name'], 'fixture:local')
        self.assertEqual(settings['digest'], 'digest')
        self.assertEqual(settings['transport'], 'fixed_local_sdk')
        self.assertEqual(settings['timeout_seconds'], client.options['timeout'])
        self.assertEqual(settings['timeout_seconds'], 45)
        self.assertEqual(settings['retries'], 0)
        self.assertEqual(settings['runtime_retrieval_settings'], {
            'model': 'fixture:runtime', 'num_predict_ceiling': 64, 'effective_num_predict': 64,
            'timeout_seconds': 17, 'retries': 2,
        })
        self.assertEqual(settings['request_settings']['options']['num_predict'], 512)
        self.assertEqual(report['model_calls'], len(client.requests))
        self.assertEqual(len(client.requests), settings['max_calls'])
        self.assertEqual(len(client.requests), 10)
        self.assertTrue(all(request['model'] == 'fixture:local' and request['format'] == 'json'
                            and request['options']['num_predict'] == 512
                            for request in client.requests))
        fresh = report['groups']['fresh']
        selector = fresh['actor_selector']
        fallback_by_id = {row['id']: row['returned_ids'] for row in fresh['actor_fallback']['results']}
        errors = [row for row in selector['results'] if row['selector_error']]
        self.assertEqual([row['id'] for row in errors], ['F05', 'F07'])
        for row in errors:
            self.assertEqual(row['selector_error'], 'TimeoutError')
            self.assertTrue(row['returned_ids'])
            self.assertEqual(row['returned_ids'], fallback_by_id[row['id']])
        metrics = selector['metrics']
        self.assertEqual(metrics['completed_measurements'], 8)
        self.assertEqual(metrics['selector_errors'], 2)
        self.assertEqual(metrics['recall_at_5'], retrieval_eval.rate(0, 5))
        self.assertEqual(metrics['mrr_at_5'], retrieval_eval.rate(0, 5))
        self.assertEqual(metrics['no_answer_false_positive_rate'], retrieval_eval.rate(0, 3))
        correction = errors[0]
        rank = correction['returned_ids'].index(correction['gold_ids'][0]) + 1
        self.assertLessEqual(rank, 5)
        self.assertEqual(metrics['delivered_results'], {
            'recall_at_5': retrieval_eval.rate(1, 6),
            'mrr_at_5': retrieval_eval.rate(1 / rank, 6),
            'no_answer_false_positive_rate': retrieval_eval.rate(1, 4),
        })
        self.assertEqual(report['guard_violations'], [])


if __name__ == '__main__':
    unittest.main()
