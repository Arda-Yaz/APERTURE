"""Behavior/measurement contracts for the original synthetic Day 1 baseline."""
import importlib.util
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from isolation_support import TestTemporaryDirectory, install_guards

install_guards()
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reliability_eval', ROOT / 'tools/reliability_eval.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def metric_row(gold=(), returned=(), kind='direct', **extra):
    return dict(gold_ids=list(gold), returned_ids=list(returned), kind=kind,
                stale_returned_ids=[], inactive_returned_ids=[], ownership_tag_violations=[],
                other_subject_ids=[], candidate_ids=None, latency_seconds=.01, error=None, **extra)


class ReliabilityEvaluationTests(unittest.TestCase):
    def test_fixture_has_stable_split_gold_and_real_cap_pressure(self):
        fixture = evaluation.build_fixture()
        self.assertEqual(fixture, evaluation.build_fixture())
        cases, records = fixture['cases'], fixture['records']
        self.assertEqual(len(cases), 48)
        self.assertEqual(len({item['id'] for item in cases}), 48)
        self.assertEqual(sum(item['split'] == 'tune' for item in cases), 12)
        self.assertEqual(sum(item['split'] == 'held_out' for item in cases), 36)
        self.assertGreater(sum(item['key'].startswith('distractor_') for item in records), 96)
        by_id = {item['id']: item for item in records}
        self.assertEqual(len(by_id), len(records))
        for item in cases:
            self.assertTrue(set(item['gold_ids']).issubset(by_id))
            if item['kind'] == 'old_zero_overlap':
                tokens = lambda text: set(re.findall(r'\w+', text.casefold()))
                self.assertFalse(tokens(item['query']) & tokens(by_id[item['gold_ids'][0]]['content']))
        self.assertEqual({item['kind'] for item in cases}, {'direct', 'paraphrase', 'cross_language',
            'no_answer', 'correction', 'forgotten', 'ownership', 'follow_up', 'old_zero_overlap'})

    def test_metrics_use_nonempty_gold_and_distinguish_no_answer_false_positives(self):
        rows = [metric_row(['a', 'b'], ['x', 'a']), metric_row(['c'], ['x']),
                metric_row([], ['x'], 'no_answer'), metric_row([], [], 'no_answer')]
        summary = evaluation.summarize(rows)
        self.assertEqual(summary['macro_recall_at_5'], dict(numerator=.5, denominator=2, rate=.25))
        self.assertEqual(summary['mrr_at_5'], dict(numerator=.5, denominator=2, rate=.25))
        self.assertEqual(summary['no_answer_false_positive_rate'], dict(numerator=1, denominator=2, rate=.5))
        self.assertEqual(summary['candidate_gold_coverage']['denominator'], 0)
        self.assertIsNone(summary['candidate_gold_coverage']['rate'])
        self.assertEqual(summary['retrieval_latency']['samples'], 4)

    def test_correction_requires_current_top5_and_no_stale_return_anywhere(self):
        good = metric_row(['new'], ['new'], 'correction')
        stale = metric_row(['new'], ['new', 'old'], 'correction')
        stale['stale_returned_ids'] = ['old']
        late = metric_row(['new'], ['a', 'b', 'c', 'd', 'e', 'new'], 'correction')
        result = evaluation.summarize([good, stale, late])
        self.assertEqual(result['current_version_accuracy'], dict(numerator=1, denominator=3, rate=1 / 3))
        self.assertEqual(result['stale_returns'], 1)

    def test_no_samples_are_unknown_instead_of_zero_quality(self):
        summary = evaluation.summarize([])
        for key in ('macro_recall_at_5', 'mrr_at_5', 'no_answer_false_positive_rate', 'current_version_accuracy'):
            self.assertIsNone(summary[key]['rate'])
        self.assertIsNone(summary['retrieval_latency']['mean_seconds'])

    def test_execution_failure_is_counted_separately_from_quality(self):
        failed = metric_row(['a'], [])
        failed['error'] = 'Synthetic measurement failure'
        summary = evaluation.summarize([metric_row(['a'], ['a']), failed])
        self.assertEqual(summary['cases'], 2)
        self.assertEqual(summary['completed_measurements'], 1)
        self.assertEqual(summary['errors'], 1)
        self.assertEqual(summary['macro_recall_at_5'], dict(numerator=1, denominator=1, rate=1))
        self.assertIsNone(evaluation.summarize([failed])['mrr_at_5']['rate'])

    def test_runtime_redirects_all_three_stores_and_preserves_inference_guard(self):
        with evaluation.isolated_runtime() as (memory, store, llm, safety):
            import experience
            import goals
            import ollama
            paths = [store.db_path, experience._STORE.db_path, goals._STORE.db_path]
            for path in paths:
                self.assertFalse(path.resolve().is_relative_to((ROOT / 'data').resolve()))
                self.assertTrue(any(path.resolve().is_relative_to(root) for root in safety._ALLOWED_ROOTS))
            self.assertIs(memory._STORE, store)
            self.assertIs(ollama.Client.chat, safety._blocked_inference)
            self.assertFalse(llm.ENABLE_DYNAMIC_SELF)
            self.assertFalse(llm.ENABLE_RELATIONSHIP)
            self.assertFalse(llm.ENABLE_SELF_CONSOLIDATION)
        self.assertFalse(any(path.exists() for path in paths))

    def test_report_measures_real_paths_and_preserves_bounded_history_coverage(self):
        with TestTemporaryDirectory() as directory:
            destination = Path(directory) / 'report.json'
            report = evaluation.evaluate(destination)
            self.assertEqual(json.loads(destination.read_text(encoding='utf-8')), report)
            self.assertEqual(len(report['results']), 144)
            self.assertEqual(report['guard_violations'], [])
            self.assertEqual(report['live_model']['status'], 'NOT RUN')
            self.assertEqual(report['full_response_latency']['samples'], 0)
            self.assertIsNone(report['unsupported_history_claims']['count'])
            self.assertEqual(report['headline_split'], 'held_out')
            for path, splits in report['aggregates'].items():
                for split, count in [('tune', 12), ('held_out', 36)]:
                    summary = splits[split]
                    self.assertEqual(summary['cases'], count)
                    self.assertEqual(summary['errors'], 0, path)
                    self.assertEqual(summary['stale_returns'], 0, path)
                    self.assertEqual(summary['inactive_returns'], 0, path)
                    self.assertEqual(summary['ownership_tag_violations'], 0, path)
                    self.assertEqual(summary['retrieval_latency']['samples'], count)
            contract = [row for row in report['results'] if row['path'] == 'mocked_selector_contract']
            old = [row for row in contract if row['kind'] == 'old_zero_overlap']
            self.assertEqual(len(old), 2)
            self.assertTrue(all(row['candidate_gold_ids'] == row['gold_ids'] == row['returned_ids']
                                for row in old))
            self.assertTrue(all(0 < row['candidate_count'] <= 96 for row in contract))
            self.assertTrue(all(row['returned_ids'] == [] for row in contract if not row['gold_ids']))
            lexical = [row for row in report['results'] if row['path'] == 'lexical_tool']
            self.assertTrue(all(row['candidate_ids'] is None for row in lexical))

    def test_production_output_rejected_before_fixture_write_or_runtime(self):
        with patch.object(evaluation, 'isolated_runtime') as runtime, patch.object(Path, 'write_text') as write:
            with self.assertRaisesRegex(ValueError, 'production data'):
                evaluation.evaluate(ROOT / 'data' / 'forbidden-evaluation.json')
            runtime.assert_not_called()
            write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
