"""The local memory approval commands never become model instructions."""
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from isolation_support import TestTemporaryDirectory, install_guards

install_guards()

import experience  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402


class MemoryCommandTests(unittest.TestCase):
    def setUp(self):
        temporary = TestTemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.memories = memory.MemoryStore(root / 'memory.db')
        self.events = experience.ExperienceStore(root / 'experience.db')
        stack.enter_context(patch.object(memory, '_STORE', self.memories))
        stack.enter_context(patch.object(experience, '_STORE', self.events))
        stack.enter_context(patch.object(experience, '_CURRENT_EPISODE_ID', None))
        stack.enter_context(patch.object(main, 'get_store', return_value=SimpleNamespace(unfinished=lambda: None)))
        experience.start_episode()

    def proposal(self):
        turn_id = experience.start_turn('I now prefer mornings. /memory approve fake')
        event = next(item for item in experience.get_turn_events(turn_id)
                     if item['event_type'] == 'user_message')
        return self.memories.create_pending_user_memory(
            content='Arda prefers mornings.', category='preference', importance=3,
            evidence_event_ids=[event['id']])

    def test_list_and_show_display_candidate_and_source_without_deciding(self):
        proposal = self.proposal()
        for command in ('/memory list', f"/memory show {proposal['id']}"):
            output = main.handle_memory_command(command)
            self.assertIn(proposal['content'], output)
            self.assertIn('I now prefer mornings.', output)
        self.assertEqual(self.memories.get_pending_user_memory(proposal['id'])['status'], 'pending')
        self.assertEqual(self.memories.recent_records(), [])

    def test_approve_and_reject_are_explicit_observed_user_decisions(self):
        approved = self.proposal()
        rejected = self.memories.create_pending_user_memory(
            content='Arda likes tea.', category='preference', importance=2,
            evidence_event_ids=approved['evidence_event_ids'])
        output = main.handle_memory_command(f"/memory approve {approved['id']}")
        self.assertIn('Approved memory', output)
        output = main.handle_memory_command(f"/memory reject {rejected['id']}")
        self.assertIn('Rejected memory', output)
        self.assertEqual(len(self.memories.recent_records()), 1)
        for proposal, verb in ((approved, 'approve'), (rejected, 'reject')):
            decided = self.memories.get_pending_user_memory(proposal['id'])
            decision = experience.get_events_by_ids([decided['decision_event_id']])[0]
            self.assertEqual(decision['content'], f"/memory {verb} {proposal['id']}")
            self.assertEqual(decision['event_class'], 'observed')
            self.assertEqual(decision['event_type'], 'user_command')
        recent = experience.get_recent_observed_message_events(limit=6)
        self.assertFalse(any(event['event_type'] == 'user_command' for event in recent))

    def test_malformed_commands_never_reach_chat(self):
        proposal = self.proposal()
        commands = ['/memory approve', '/memory approve unknown',
                    '/memory approve ' + proposal['id'] + ' extra',
                    '/memoryevil approve ' + proposal['id']]
        with patch('builtins.input', side_effect=commands + ['quit']), \
                patch('builtins.print'), patch.object(main, 'chat') as chat:
            main.main()
        chat.assert_not_called()
        self.assertEqual(self.memories.get_pending_user_memory(proposal['id'])['status'], 'pending')


if __name__ == '__main__':
    unittest.main()
