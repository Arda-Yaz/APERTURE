import sys
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
import isolation_support as safety
safety.install_guards()
import task_controller as controller


def pair(tool='read_source', status='ok', index=0, content=None):
    call = {'id': f'call{index}', 'seq': index * 2 + 1, 'turn_id': 'turn', 'episode_id': 'episode',
            'event_class': 'observed', 'event_type': 'tool_call',
            'metadata': {'tool_name': tool, 'arguments': {'root': 'fixture', 'path': 'a.py', 'secret': '[REDACTED]'}}}
    result = {'id': f'result{index}', 'seq': index * 2 + 2, 'turn_id': 'turn', 'episode_id': 'episode',
              'event_class': 'observed', 'event_type': 'tool_result',
              'content': content if content is not None else json.dumps({'status': 'ok', 'path': 'a.py',
                   'range': {'returned_start': 2}, 'coverage': {'complete': False}, 'truncated': True,
                   'continuation': 'next', 'evidence': [{'path': 'a.py', 'line': 2, 'text': 'class Store: pass'}]}),
              'metadata': {'tool_name': tool, 'status': status}}
    return call, result


def verdict(**changes):
    value = {'status': 'satisfied', 'rationale': 'The definition is observed.',
             'evidence_refs': ['result0'], 'missing_requirements': []}
    value.update(changes)
    return json.dumps(value)


class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.evidence = controller.EvidenceBundle()
        self.evidence.add(*pair())

    def test_strict_schema_and_full_json(self):
        invalid = ['{"done":"false"}', '{"done":1}', '```json\n' + verdict() + '\n```',
                   verdict() + ' trailing', '{', 'null', '[]',
                   verdict(status=True), verdict(status=1), verdict(status='done'),
                   verdict(rationale=''), verdict(rationale='x' * 501), verdict(extra=True),
                   verdict(evidence_refs='result0'), verdict(evidence_refs=['result0', 'result0']),
                   verdict(evidence_refs=['invented']), verdict(evidence_refs=[]),
                   verdict(missing_requirements=['unfinished']), verdict(status='incomplete'),
                   verdict(status='incomplete', missing_requirements=[True]),
                   verdict(status='incomplete', missing_requirements=['x'] * 9),
                   verdict(status='incomplete', missing_requirements=['x' * 241]),
                   verdict().replace('"status": "satisfied"', '"status": "incomplete", "status": "satisfied"')]
        for output in invalid:
            with self.subTest(output=output):
                self.assertEqual(controller.parse_assessment(output, self.evidence).status, 'invalid')
        self.assertEqual(controller.parse_assessment(verdict(), self.evidence).status, 'satisfied')
        self.assertEqual(controller.parse_assessment(verdict(status='incomplete', missing_requirements=['Read remaining lines.']), self.evidence).status, 'incomplete')

    def test_ineligible_references_and_unpaired_events(self):
        for tool, status in [('checkpoint_goal', 'ok'), ('recall_recent_activity', 'ok'),
                             ('search_memory', 'ok'), ('read_source', 'error'), ('read_source', 'permission_denied')]:
            evidence = controller.EvidenceBundle()
            evidence.add(*pair(tool, status))
            self.assertEqual(controller.parse_assessment(verdict(), evidence).status, 'invalid')
        for field, value in [('event_class', 'derived'), ('event_type', 'assistant_message'),
                             ('seq', 500), ('turn_id', 'other'), ('episode_id', 'other')]:
            evidence = controller.EvidenceBundle()
            call, result = pair()
            result[field] = value
            evidence.add(call, result)
            self.assertFalse(evidence.supplied_refs)
        evidence.add(None, pair()[1])
        self.assertFalse(evidence.eligible_refs)

    def test_bounded_evidence_and_evicted_refs(self):
        for index in range(1, 20):
            self.evidence.add(*pair(index=index))
        self.assertLessEqual(len(self.evidence.entries), 12)
        self.assertLessEqual(len(json.dumps(self.evidence.entries, ensure_ascii=False)), 16000)
        self.assertEqual(controller.parse_assessment(verdict(), self.evidence).status, 'invalid')
        self.assertGreater(self.evidence.omitted, 0)

    def test_partial_positive_truncation_metadata_and_model_boundary(self):
        with patch.object(controller, 'ollama_chat', return_value=SimpleNamespace(message=SimpleNamespace(content=verdict()))) as model:
            result = controller.assess_task('Locate Store definition', 'a.py:2 defines Store', self.evidence,
                                            reports={'progress': 'claimed done', 'next_action': 'verify'})
        self.assertEqual(result.status, 'satisfied')
        payload = json.loads(model.call_args.kwargs['messages'][1]['content'])
        observation = payload['evidence']['entries'][0]['evidence']
        self.assertEqual(observation['inspection']['range']['returned_start'], 2)
        self.assertFalse(observation['inspection']['coverage']['complete'])
        self.assertTrue(observation['inspection']['truncated'])
        self.assertEqual(observation['inspection']['continuation'], 'next')
        self.assertEqual(payload['unverified_reports_and_plans']['next_action'], 'verify')
        self.assertIn('Partial searches do not prove absence', model.call_args.kwargs['messages'][0]['content'])
        self.assertNotIn('secret', observation['arguments'])

    def test_inference_failure_and_missing_response_fail_closed(self):
        for value in [RuntimeError('offline'), SimpleNamespace(message=SimpleNamespace(content=None))]:
            with patch.object(controller, 'ollama_chat', **({'side_effect': value} if isinstance(value, Exception) else {'return_value': value})):
                self.assertNotEqual(controller.assess_task('goal', 'done', self.evidence).status, 'satisfied')
        with patch.object(controller, 'ollama_chat', return_value=SimpleNamespace(message=SimpleNamespace(content=verdict()))):
            self.assertFalse(controller.is_task_complete('goal', 'done', 'raw content'))


if __name__ == '__main__':
    unittest.main()
