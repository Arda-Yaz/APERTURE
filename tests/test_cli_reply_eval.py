import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from isolation_support import install_guards


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'app'), str(ROOT / 'tools')]
install_guards()
import cli_reply_eval
import llm
import task_controller
from inference import InferenceError


class CliReplyEvaluationTests(unittest.TestCase):
    def test_historical_controls_accept_only_the_supplied_reference(self):
        self.assertEqual(cli_reply_eval.history_controls(llm), {
            'grounded_history': True, 'invented_history_blocked': True,
        })

    def test_file_role_audit_counts_semantic_downgrade_that_allows_a_claim(self):
        def classify(should_pass, roles, refs, *, messages, **request):
            goal = json.loads(messages[-1]['content'])['objective']
            relabel = not should_pass and goal == 'Read a.txt and b.txt.'
            payload = {'status': 'satisfied' if should_pass or relabel else 'insufficient_evidence',
                       'rationale': 'Synthetic controller result.', 'evidence_refs': refs,
                       'missing_requirements': [], 'file_requirements': dict(roles)}
            if relabel:
                payload['file_requirements']['b.txt'] = 'context'
            return SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))

        result = cli_reply_eval.measure_file_roles(classify, task_controller)
        self.assertEqual(result['metrics']['false_accept'], {'count': 1, 'total': 4})
        self.assertEqual(result['metrics']['required_labels_downgraded'], {'count': 1, 'total': 11})

    def test_unavailable_file_assessment_is_reported_separately_from_missing_labels(self):
        def classify(*fixture_values, **request):
            raise InferenceError('timeout', role='task_controller')

        result = cli_reply_eval.measure_file_roles(classify, task_controller)
        self.assertEqual(result['metrics']['request_errors'], 6)
        self.assertEqual(result['metrics']['required_labels_missing'], {'count': 0, 'total': 0})
        self.assertTrue(all(row['status'] == 'unavailable' for row in result['cases']))
