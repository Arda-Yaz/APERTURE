"""Synthetic memory acceptance runner: no live model or production stores."""

import json
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from isolation_support import TestTemporaryDirectory, install_guards

install_guards()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'app'))
import memory_acceptance_eval as evaluator
import reflection


def response(payload):
    return SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))


def sample_case(decision, seed=None):
    case = {
        'id': 'synthetic_' + decision, 'split': 'dev',
        'failure_class': 'temporal_update',
        'messages': [
            {'role': 'assistant', 'content': 'This is a synthetic assistant example about choosing a schedule.'},
            {'role': 'user', 'content': 'My durable personal preference is to work in the morning, because I can focus before other commitments.'},
        ],
        'expected': {'decision': decision},
        'review_criterion': 'The human reviewer checks ownership and wording.',
    }
    if seed:
        case['seed_memory'] = {'content': 'The user prefers evening work.',
                               'category': 'preference', 'subject': 'user'}
        case['seed_observation'] = 'For my own schedule, I prefer evening work.'
    return case


def fake_model(decision):
    calls = []

    def call(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) % 2:
            return response({'candidate': None if decision == 'reject' else {
                'content': 'The user prefers morning work.',
                'category': 'preference', 'importance': 3}})
        candidate = None
        if decision != 'reject':
            candidate = {'content': 'The user prefers morning work.',
                         'category': 'preference', 'importance': 3,
                         'needs_confirmation': decision == 'pending'}
            if decision == 'supersede':
                from memory import _get_store
                candidate['supersedes_memory_id'] = _get_store().recent_records()[0]['id']
        return response({'user_memory': candidate, 'self_memory': None})

    return call, calls


class MemoryAcceptanceEvaluationTests(unittest.TestCase):
    def test_fixture_and_dry_run_select_dev_without_importing_app(self):
        fixture = evaluator.load_fixture()
        self.assertEqual(len(fixture['cases']), 30)
        with TestTemporaryDirectory() as directory:
            output = Path(directory) / 'dry.json'
            with patch.object(evaluator.importlib, 'import_module',
                              side_effect=AssertionError('dry run imported app')):
                report = evaluator.evaluate(output, split='dev', live=False)
            self.assertEqual(report['selected_split'], 'dev')
            self.assertEqual(len(report['prepared_cases']), 10)
            self.assertTrue(all(case['split'] == 'dev'
                                for case in report['prepared_cases']))
            self.assertEqual(report['cases'], [])
            self.assertEqual(json.loads(output.read_text(encoding='utf-8')),
                             report)
            self.assertEqual(report['fixture_sha256'], evaluator._digest(evaluator.FIXTURE))

    def test_dry_run_allows_absent_optional_inference_source(self):
        with TestTemporaryDirectory() as directory:
            missing = Path(directory) / 'absent_in_clean_checkout.py'
            output = Path(directory) / 'prepared.json'
            with patch.object(evaluator, 'OPTIONAL_INFERENCE_SOURCE', missing):
                report = evaluator.evaluate(output, live=False, split='dev')
            self.assertEqual(report['source_sha256']['app/inference.py'], None)
            self.assertEqual(len(report['prepared_cases']), 10)
            self.assertEqual(len(report['source_sha256']['app/reflection.py']), 64)
            self.assertEqual(json.loads(output.read_text(encoding='utf-8')),
                             report)

    def test_fixture_rejects_seed_without_observed_user_speech(self):
        fixture = evaluator.load_fixture()
        seeded = next(case for case in fixture['cases'] if 'seed_memory' in case)
        seeded.pop('seed_observation')
        with TestTemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.json'
            path.write_text(json.dumps(fixture), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'observed user source'):
                evaluator.load_fixture(path)

    def test_real_write_path_with_fake_create_reject_pending_and_supersede(self):
        results = []
        for decision in ('create', 'reject', 'pending', 'supersede'):
            with self.subTest(decision=decision), TestTemporaryDirectory() as directory:
                model, calls = fake_model(decision)
                budget = {'calls': 0, 'max_calls': 2, 'started': time.perf_counter(),
                          'max_seconds': 30}
                with patch.object(reflection, 'REQUIRE_REFLECTION_USER_MEMORY_APPROVAL', False):
                    result = evaluator.run_case(sample_case(decision,
                        seed=decision == 'supersede'), Path(directory),
                        model_call=model, budget=budget)
                self.assertIsNone(result['error'])
                self.assertEqual(result['observed_decision'], decision)
                self.assertTrue(result['decision_match'])
                self.assertTrue(result['provenance_valid'])
                self.assertTrue(result['lineage_valid'])
                self.assertEqual(budget['calls'], 2)
                self.assertEqual(len(calls), 2)
                self.assertEqual(len(result['model_calls']), 2)
                self.assertEqual(result['semantic_verdict'], 'unreviewed')
                self.assertEqual(result['observed_events'][0]['actor'], 'aperture')
                self.assertEqual(result['observed_events'][0]['event_type'],
                                 'assistant_message')
                self.assertEqual(result['observed_events'][1]['actor'], 'arda')
                if decision == 'supersede':
                    seed_event = result['seed_observation_event']
                    self.assertEqual(seed_event['content'],
                                     sample_case(decision, seed=True)['seed_observation'])
                    self.assertEqual(seed_event['actor'], 'arda')
                    self.assertNotEqual(seed_event['episode_id'],
                                        result['observed_events'][0]['episode_id'])
                if decision == 'pending':
                    self.assertEqual(len(result['pending']), 1)
                    self.assertEqual(result['after_memories'], [])
                results.append(result)
        metrics = evaluator.summarize(results)['dev']
        self.assertEqual(metrics['create_capture'], 1)
        self.assertEqual(metrics['reject_specificity'], 1)
        self.assertEqual(metrics['pending_capture'], 1)
        self.assertEqual(metrics['supersession_capture'], 1)
        self.assertEqual(metrics['false_active_supersessions'], 0)
        self.assertEqual(metrics['pending_proposals'], 1)
        self.assertEqual(metrics['cases_with_pending_proposals'], 1)
        self.assertEqual(metrics['semantic_reviewed'], 0)
        self.assertEqual(metrics['semantic_reviewable'], 3)
        self.assertIsNone(metrics['semantic_fidelity'])

    def test_policy_modes_measure_false_supersession_and_pending_burden(self):
        case = sample_case('reject', seed=True)
        outcomes = []
        for approval in (False, True):
            with self.subTest(approval=approval), TestTemporaryDirectory() as directory:
                budget = {'calls': 0, 'max_calls': 2, 'started': time.perf_counter(),
                          'max_seconds': 30}
                with patch.object(reflection, 'REQUIRE_REFLECTION_USER_MEMORY_APPROVAL', approval):
                    result = evaluator.run_case(case, Path(directory),
                                                model_call=fake_model('supersede')[0],
                                                budget=budget)
                self.assertIsNone(result['error'])
                seed, = result['before_memories']
                retained = next(record for record in result['after_memories']
                                if record['id'] == seed['id'])
                self.assertEqual(retained['active'], approval)
                self.assertEqual(result['observed_decision'], 'pending' if approval else 'supersede')
                metrics = evaluator.summarize([result])['dev']
                outcomes.append(tuple(metrics[name] for name in (
                    'false_active_writes', 'false_active_supersessions', 'pending_proposals',
                )))
        self.assertEqual(outcomes, [(1, 1, 0), (0, 0, 1)])

    def test_offline_fidelity_score_preserves_raw_run_and_decision_score(self):
        rows = [
            dict(sample_case('create'), status='complete', decision_match=True,
                 observed_decision='create', semantic_verdict='unreviewed',
                 after_memories=[{'id': 'new', 'subject': 'user', 'active': True,
                                  'content': 'The user prefers technical explanations in Turkish.'}]),
            dict(sample_case('reject'), id='negative', status='complete',
                 decision_match=True, observed_decision='reject', semantic_verdict='unreviewed'),
        ]
        report = {'live': True, 'fixture_sha256': evaluator._digest(evaluator.FIXTURE),
                  'model': 'synthetic', 'cases': rows}
        with TestTemporaryDirectory() as directory:
            root = Path(directory)
            source, review_file, output = (root / name for name in ('scored.json.tmp', 'review.json', 'scored.json'))
            source.write_text(json.dumps(report), encoding='utf-8')
            review = {'report_sha256': evaluator._digest(source), 'cases': {
                rows[0]['id']: {'verdict': 'unfaithful', 'reason': 'English code identifier exception omitted.'},
            }}
            judgment = review['cases'][rows[0]['id']]
            judgment['checks'] = dict.fromkeys(evaluator.SEMANTIC_DIMENSIONS, 'not_applicable')
            judgment['checks']['exception'] = 'lost'
            review['reviewer'] = 'agent'
            review_file.write_text(json.dumps(review), encoding='utf-8')
            with patch.object(evaluator.importlib, 'import_module',
                              side_effect=AssertionError('offline scoring imported app')):
                scored = evaluator.score_semantic_review(source, review_file, output)
            metrics = scored['metrics']['dev']
            self.assertEqual(metrics['create_capture'], 1)
            self.assertEqual(metrics['semantic_reviewable'], 1)
            self.assertEqual(metrics['semantic_reviewed'], 1)
            self.assertEqual(metrics['semantic_fidelity'], 0)
            self.assertEqual(scored['model'], report['model'])
            self.assertEqual(scored['fixture_sha256'], report['fixture_sha256'])
            self.assertEqual(scored['cases'][0]['after_memories'], rows[0]['after_memories'])
            self.assertEqual(scored['cases'][0]['semantic_checks'], judgment['checks'])
            self.assertEqual(scored['semantic_reviewer'], 'agent')
            self.assertEqual(evaluator._digest(source), review['report_sha256'])
            review['report_sha256'] = 'wrong run'
            review_file.write_text(json.dumps(review), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'exact evaluated report'):
                evaluator.score_semantic_review(source, review_file, output, overwrite=True)
            review.update(report_sha256=evaluator._digest(output), reviewer='human', cases={})
            review_file.write_text(json.dumps(review), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'raw, unreviewed report'):
                evaluator.score_semantic_review(output, review_file, root / 'relabeled.json')

    def test_failed_model_attempt_is_counted_and_reported(self):
        with TestTemporaryDirectory() as directory:
            def fail(*args, **kwargs):
                raise OSError('synthetic model failure')

            budget = {'calls': 0, 'max_calls': 1, 'started': time.perf_counter(),
                      'max_seconds': 30}
            result = evaluator.run_case(sample_case('reject'), Path(directory),
                                        model_call=fail, budget=budget)
            self.assertEqual(budget['calls'], 1)
            self.assertEqual(result['error'], 'OSError')
            self.assertEqual(result['model_calls'][0]['error'], 'OSError')
            self.assertFalse(result['decision_match'])

    def test_live_injected_model_saves_incremental_report(self):
        with TestTemporaryDirectory() as directory:
            output = Path(directory) / 'live.json'
            model, calls = fake_model('reject')
            snapshots = []

            def inspect_during_call(*args, **kwargs):
                snapshots.append(json.loads(output.read_text(encoding='utf-8')))
                return model(*args, **kwargs)

            report = evaluator.evaluate(output, live=True, split='dev',
                                        max_cases=1, max_calls=2,
                                        model_call=inspect_during_call,
                                        temporary_factory=TestTemporaryDirectory)
            self.assertTrue(report['complete'])
            self.assertEqual(report['selected_split'], 'dev')
            self.assertEqual(report['model_calls_attempted'], 2)
            self.assertEqual(len(calls), 2)
            self.assertEqual(snapshots[0]['model_calls_attempted'], 1)
            self.assertIsNone(snapshots[0]['cases'][0]['model_calls'][0]['latency_seconds'])
            self.assertEqual(len(report['cases']), 1)
            self.assertEqual(report['cases'][0]['status'], 'complete')
            self.assertEqual(json.loads(output.read_text(encoding='utf-8')),
                             report)

    def test_live_metadata_and_model_calls_use_closed_loopback_clients(self):
        with TestTemporaryDirectory() as directory, \
                patch.dict(os.environ, {'APERTURE_OLLAMA_HOST': 'http://127.0.0.1:11434'}), \
                patch('ollama.Client') as constructor:
            client = constructor.return_value.__enter__.return_value
            client.list.return_value.model_dump.return_value = {
                'models': [{'model': reflection.MODEL, 'digest': 'test-model-digest'}],
            }
            client.chat.side_effect = fake_model('reject')[0]
            report = evaluator.evaluate(
                Path(directory) / 'local.json', live=True, split='dev',
                max_cases=1, max_calls=2, timeout_seconds=5,
                temporary_factory=TestTemporaryDirectory,
            )
            self.assertTrue(report['complete'])
            self.assertEqual(report['model_metadata']['digest'], 'test-model-digest')
            self.assertEqual(client.list.call_count, 1)
            self.assertEqual(client.chat.call_count, 2)
            self.assertEqual(constructor.call_count, 3)
            self.assertEqual(constructor.return_value.__exit__.call_count, 3)
            for item in constructor.call_args_list:
                self.assertEqual(item.kwargs, {
                    'host': 'http://127.0.0.1:11434', 'timeout': 5,
                    'follow_redirects': False, 'trust_env': False,
                })

    def test_policy_reports_keep_fixture_source_and_model_options_equal(self):
        with TestTemporaryDirectory() as directory:
            reports = []
            for approval in (False, True):
                with patch.object(reflection, 'REQUIRE_REFLECTION_USER_MEMORY_APPROVAL', approval):
                    reports.append(evaluator.evaluate(
                        Path(directory) / f'{approval}.json', live=True, split='dev',
                        max_cases=1, max_calls=2, model_call=fake_model('create')[0],
                        temporary_factory=TestTemporaryDirectory,
                    ))
            automatic, queued = reports
            self.assertTrue(all(report['complete'] for report in reports))
            for key in ('fixture_sha256', 'source_sha256', 'model', 'model_settings', 'limits'):
                self.assertEqual(automatic[key], queued[key], key)
            self.assertFalse(automatic['require_reflection_user_memory_approval'])
            self.assertTrue(queued['require_reflection_user_memory_approval'])
            self.assertEqual(automatic['cases'][0]['observed_decision'], 'create')
            self.assertEqual(queued['cases'][0]['observed_decision'], 'pending')
            self.assertEqual(automatic['cases'][0]['model_calls'][1]['response'],
                             queued['cases'][0]['model_calls'][1]['response'])

    def test_live_failure_remains_in_incremental_report(self):
        with TestTemporaryDirectory() as directory:
            output = Path(directory) / 'failure.json'

            def fail(*args, **kwargs):
                raise OSError('synthetic model failure')

            report = evaluator.evaluate(output, live=True, split='dev',
                                        max_cases=1, max_calls=1,
                                        model_call=fail,
                                        temporary_factory=TestTemporaryDirectory)
            self.assertFalse(report['complete'])
            self.assertEqual(report['model_calls_attempted'], 1)
            self.assertEqual(report['errors'], [
                {'case_id': report['cases'][0]['id'], 'type': 'OSError'}])
            self.assertEqual(json.loads(output.read_text(encoding='utf-8')),
                             report)

    def test_reject_nonlocal_host_and_production_paths(self):
        for host in ('https://remote.example:11434',
                     'http://user:pass@localhost:11434',
                     'file:///localhost/ollama'):
            with self.subTest(host=host), self.assertRaises(ValueError):
                evaluator._localhost_host(host)
        with self.assertRaises(ValueError):
            evaluator.evaluate(ROOT / 'data' / 'evaluation.json')
        with self.assertRaises(ValueError):
            with evaluator._isolated_stores(ROOT / 'data' / 'evaluation'):
                pass

    def test_existing_output_requires_explicit_overwrite(self):
        with TestTemporaryDirectory() as directory:
            output = Path(directory) / 'sentinel.json'
            output.write_text('leave me alone', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                evaluator.evaluate(output, split='dev')
            self.assertEqual(output.read_text(encoding='utf-8'), 'leave me alone')
            report = evaluator.evaluate(output, split='dev', overwrite=True)
            self.assertEqual(len(report['prepared_cases']), 10)
            self.assertEqual(json.loads(output.read_text(encoding='utf-8')),
                             report)

    def test_cli_rejects_output_outside_scratch(self):
        with TestTemporaryDirectory(prefix='memory-eval-output-',
                                    dir=ROOT / 'tests') as directory:
            output = Path(directory) / 'report.json'
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                evaluator.main(['--output', str(output)])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())

    def test_total_deadline_reduces_and_exhausts_call_timeout(self):
        budget = {'max_seconds': 10, 'started': time.perf_counter() - 8}
        self.assertLess(evaluator._remaining_timeout(budget, 30), 3)
        budget['started'] = time.perf_counter() - 11
        with self.assertRaises(evaluator.EvaluationBudgetError):
            evaluator._remaining_timeout(budget, 30)


if __name__ == '__main__':
    unittest.main()
