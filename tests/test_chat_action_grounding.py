import io
import json
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from isolation_support import TestTemporaryDirectory, install_guards

install_guards()
import experience
import goals
import llm
import main
import memory
import task_controller
from inference import InferenceError


def reply(content='', calls=()):
    return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[
        SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))
        for name, arguments in calls
    ]))


def verdict(status, refs=()):
    return reply(json.dumps({
        'status': status, 'rationale': 'Synthetic assessment.',
        'evidence_refs': list(refs), 'missing_requirements': [],
    }))


class ChatActionGroundingTests(unittest.TestCase):
    claim = 'The synthetic fixture was read successfully.'
    read_call = ('read_file', {'path': 'fixture.txt'})

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        directory = Path(stack.enter_context(TestTemporaryDirectory()))
        for module, value in (
            (memory, memory.MemoryStore(directory / 'memory.db')),
            (experience, experience.ExperienceStore(directory / 'experience.db')),
            (goals, goals.GoalStore(directory / 'goals.db')),
        ):
            stack.enter_context(patch.object(module, '_STORE', value))
        stack.enter_context(patch.object(experience, '_CURRENT_EPISODE_ID', None))
        stack.enter_context(patch.object(llm, 'maybe_reflect', return_value=None))
        self.read = Mock(return_value='Original synthetic file content')
        stack.enter_context(patch.dict(llm.TOOL_MAP, {'read_file': self.read}))
        self.permission = stack.enter_context(patch.object(llm, 'check_permission', return_value=True))

    def run_cli(self, inputs, responses, controller):
        with patch('builtins.input', side_effect=[*inputs, 'exit']), \
                patch('sys.stdout', new_callable=io.StringIO) as output, \
                patch.object(llm, 'ollama_chat', side_effect=responses), \
                patch.object(task_controller, 'ollama_chat', side_effect=controller) as judge:
            main.main()
        events = experience.get_episode_events(experience.get_latest_episode_id())
        answers = [event['content'] for event in events if event['event_type'] == 'assistant_message']
        for answer in answers:
            self.assertIn(answer, output.getvalue())
        return answers, events, judge

    def run_read(self, controller):
        answers, events, judge = self.run_cli(
            ['Read fixture.txt'], [reply(calls=[self.read_call]), reply(self.claim)], controller)
        answer, = answers
        outcome, = [event for event in events if event['event_type'] == 'tool_result']
        return answer, outcome, judge

    @staticmethod
    def satisfy_with_recorded_ref(*, messages, **kwargs):
        payload = json.loads(messages[-1]['content'])
        entry, = payload['evidence']['entries']
        return verdict('satisfied', [entry['evidence']['result_event_id']])

    def test_current_success_uses_positive_observed_result(self):
        answer, outcome, judge = self.run_read(self.satisfy_with_recorded_ref)
        self.assertIn(self.claim, answer)
        self.assertIn('Current turn tool observations', answer)
        self.assertIn('tool reported success', answer)
        self.assertIn(outcome['id'], answer)
        self.assertEqual(outcome['metadata']['status'], 'ok')
        self.read.assert_called_once_with(path='fixture.txt')
        judge.assert_called_once()

    def test_conversation_label_cannot_hide_unobserved_file_completion(self):
        for target, assessment in [('fixture.txt', reply('action_outcome')),
                                   ('docs/README', reply('action_outcome')),
                                   ('fixture.txt', InferenceError('timeout', role='task_controller'))]:
            with self.subTest(target=target, assessment=assessment):
                answers, _, judge = self.run_cli(
                    [f'Read {target}'], [reply(f'{llm.CONVERSATION_PREFIX} {self.claim}')], [assessment],
                )
                self.assertEqual(answers, [llm.UNVERIFIED_ACTION])
                judge.assert_called_once()
        self.read.assert_not_called()

    def test_conversation_about_a_named_file_can_pass_without_a_tool(self):
        answer = 'That is a plain text filename.'
        answers, _, judge = self.run_cli(
            ['What kind of name is fixture.txt?'], [reply(f'{llm.CONVERSATION_PREFIX} {answer}')],
            [reply('conversation')],
        )
        self.assertEqual(answers, [answer])
        self.read.assert_not_called()
        judge.assert_called_once()

    def test_success_of_a_different_file_does_not_verify_completion(self):
        for target, wrong_path in [('fixture.txt', 'other.txt'), ('./fixture.txt', 'other/fixture.txt')]:
            with self.subTest(target=target, wrong_path=wrong_path):
                answers, events, judge = self.run_cli(
                    [f'Read {target}'], [reply(calls=[('read_file', {'path': wrong_path})]), reply(self.claim)],
                    self.satisfy_with_recorded_ref,
                )
                answer, = answers
                observation, = [event for event in events if event['event_type'] == 'tool_result']
                self.assertNotIn(self.claim, answer)
                self.assertIn('could not verify completion', answer)
                self.assertIn(f'read_file "{wrong_path}": tool reported success', answer)
                self.assertIn(observation['id'], answer)
                judge.assert_called_once()

    def test_permission_denial_cannot_support_success(self):
        self.permission.return_value = False
        answer, outcome, _ = self.run_read(self.satisfy_with_recorded_ref)
        self.assertNotIn(self.claim, answer)
        self.assertIn('permission denied', answer)
        self.assertIn(outcome['id'], answer)
        self.read.assert_not_called()

    def test_tool_error_overrides_confident_actor_answer(self):
        self.read.return_value = 'READ_ERROR: synthetic failure'
        answer, outcome, _ = self.run_read([verdict('insufficient_evidence')])
        self.assertNotIn(self.claim, answer)
        self.assertIn('tool error', answer)
        self.assertIn(outcome['id'], answer)
        self.assertEqual(outcome['metadata']['status'], 'error')

    def test_unconfirmed_completion_preserves_tool_status_without_claim(self):
        for response in (
            verdict('insufficient_evidence'),
            reply('Malformed assessment'),
            InferenceError('timeout', role='task_controller'),
        ):
            with self.subTest(response=response):
                answer, outcome, judge = self.run_read([response])
                self.assertNotIn(self.claim, answer)
                self.assertIn('could not verify completion', answer)
                self.assertIn('tool reported success', answer)
                self.assertIn(outcome['id'], answer)
                judge.assert_called_once()

    def test_previous_assistant_claim_does_not_supply_a_result(self):
        experience.start_episode()
        previous_turn = experience.start_turn('Read fixture.txt')
        experience.record_assistant_message(previous_turn, self.claim)
        experience.end_episode()

        actor_replies = iter((self.claim, f'{llm.ACTION_STATUS_PREFIX} unverified\n{self.claim}'))

        def actor(*, messages, **kwargs):
            system = messages[0]['content']
            self.assertIn(self.claim, system)
            self.assertNotIn('<RECENT_OBSERVED_TOOLS>', system)
            self.assertIn(llm.ACTION_STATUS_PREFIX, system)
            return reply(next(actor_replies))

        with patch.object(llm, 'ollama_chat', side_effect=actor) as actor_call, \
                patch('builtins.input', side_effect=['Did you read fixture.txt?', 'Did it work?', 'exit']), \
                patch('sys.stdout', new_callable=io.StringIO) as output, \
                patch.object(task_controller, 'ollama_chat') as judge:
            main.main()
        self.assertNotIn(self.claim, output.getvalue())
        self.assertEqual(output.getvalue().count('No matching tool result was recorded for this turn'), 2)
        judge.assert_not_called()
        self.assertEqual(actor_call.call_count, 2)
        self.read.assert_not_called()

    def test_historical_observation_is_marked_earlier_and_not_reverified(self):
        def actor(*, messages, **kwargs):
            request = [message['content'] for message in messages if message['role'] == 'user'][-1]
            if request == 'Read fixture.txt':
                return reply(self.claim) if messages[-1]['role'] == 'tool' else reply(calls=[self.read_call])
            context = messages[0]['content'].split('<RECENT_OBSERVED_TOOLS>\n', 1)[1]
            observation, = json.loads(context.splitlines()[1])
            return reply(f"{llm.ACTION_STATUS_PREFIX} {observation['result_event_id']}")

        answers, events, judge = self.run_cli(
            ['Read fixture.txt', 'Did the earlier read succeed?'], actor, self.satisfy_with_recorded_ref)
        _, follow_up = answers
        observation, = [event for event in events if event['event_type'] == 'tool_result']
        self.assertIn('Earlier tool observation (current state unverified)', follow_up)
        self.assertIn(observation['id'], follow_up)
        self.assertIn('tool reported success', follow_up)
        self.assertNotIn('Current turn tool observations', follow_up)
        self.read.assert_called_once()
        judge.assert_called_once()

    def test_ordinary_conversation_does_not_invoke_controller(self):
        def actor_reply(*, messages, **kwargs):
            self.assertIn(llm.CONVERSATION_PREFIX, messages[0]['content'])
            return reply(f'{llm.CONVERSATION_PREFIX} Hello.')

        actor = Mock(side_effect=actor_reply)
        requests = ['Hello', 'What changed in Python 3.14?', 'What is 3/4 as a percentage?', 'Is 11/19/2026 a date?']
        answers, _, judge = self.run_cli(requests, actor, [])
        self.assertEqual(answers, ['Hello.'] * len(requests))
        self.read.assert_not_called()
        self.assertEqual(actor.call_count, len(requests))
        judge.assert_not_called()

    def test_direct_chat_keeps_plain_conversational_output(self):
        with patch.object(llm, 'ollama_chat', return_value=reply('Hello.')) as actor, \
                patch.object(task_controller, 'ollama_chat') as judge:
            self.assertEqual(llm.chat([{'role': 'user', 'content': 'Hello'}]), 'Hello.')
        actor.assert_called_once()
        judge.assert_not_called()

    def test_context_tools_keep_conversation_reply_type(self):
        search = Mock(return_value='Synthetic retrieved memory')
        recall = Mock(return_value='Synthetic recalled activity')
        with patch.dict(llm.TOOL_MAP, {'search_memory': search, 'recall_recent_activity': recall}):
            answers, _, judge = self.run_cli(
                ['What do you remember about hexagons?'], [
                    reply(calls=[('search_memory', {'query': 'hexagons'}), ('recall_recent_activity', {})]),
                    reply(f'{llm.CONVERSATION_PREFIX} Here is the context I found.'),
                ], [],
            )
        self.assertEqual(answers, ['Here is the context I found.'])
        search.assert_called_once_with(query='hexagons')
        recall.assert_called_once_with()
        judge.assert_not_called()

    def test_missing_memory_id_never_claims_removal(self):
        claim = 'The synthetic memory was removed.'
        answers, events, judge = self.run_cli(
            ['Forget memory missing_fixture'], [
                reply(calls=[('forget_memory', {'memory_id': 'missing_fixture'})]), reply(claim),
            ], [],
        )
        outcome, = [event for event in events if event['event_type'] == 'tool_result']
        answer, = answers
        self.assertEqual(outcome['metadata']['status'], 'not_found')
        self.assertIn('memory not found', answer)
        self.assertIn(outcome['id'], answer)
        self.assertNotIn(claim, answer)
        judge.assert_not_called()

    def test_pending_memory_and_current_action_status_are_both_visible(self):
        claim = 'The memory was saved and the synthetic file was read.'
        answers, events, judge = self.run_cli(
            ['Remember that I prefer hexagons and read fixture.txt'], [
                reply(calls=[('save_memory', {'content': 'The user prefers hexagons.'}), self.read_call]),
                reply(claim),
            ], [verdict('insufficient_evidence')],
        )
        answer, = answers
        read_result = next(event for event in events if event['event_type'] == 'tool_result'
                           and event['metadata']['tool_name'] == 'read_file')
        self.assertNotIn(claim, answer)
        self.assertIn('pending review', answer)
        self.assertIn('/memory approve', answer)
        self.assertIn('tool reported success', answer)
        self.assertIn(read_result['id'], answer)
        judge.assert_called_once()
