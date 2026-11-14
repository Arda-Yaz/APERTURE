"""Explicit user decisions on proposed memories preserve provenance and history."""
import sqlite3
import sys
import unittest
from contextlib import closing
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from isolation_support import TestTemporaryDirectory, install_guards

install_guards()

import experience  # noqa: E402
import memory  # noqa: E402


class PendingMemoryDecisionTests(unittest.TestCase):
    def setUp(self):
        temporary = TestTemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.memories = memory.MemoryStore(root / "memory.db")
        self.events = experience.ExperienceStore(root / "experience.db")
        stack.enter_context(patch.object(experience, "_STORE", self.events))
        stack.enter_context(patch.object(experience, "_CURRENT_EPISODE_ID", None))

    def user_event(self, content="I prefer morning tea."):
        turn_id = experience.start_turn(content)
        return next(event for event in experience.get_turn_events(turn_id)
                    if event["event_type"] == "user_message")

    def proposal(self, *, supersedes=None):
        event = self.user_event()
        proposal = self.memories.create_pending_user_memory(
            content="The user prefers morning tea.", category="preference",
            importance=4, evidence_event_ids=[event["id"]],
            supersedes_memory_id=supersedes,
        )
        return proposal, event

    def command_event(self, action, proposal_id):
        return experience.record_event(
            event_type="user_command", event_class="observed", actor="arda",
            content=f"/memory {action} {proposal_id}",
        )

    def test_approval_after_restart_forms_only_one_user_memory(self):
        proposal, event = self.proposal()
        self.assertEqual(self.memories.recent_records(), [])
        experience.end_episode()
        experience.start_episode()
        reopened = memory.MemoryStore(self.memories.db_path)

        command = self.command_event("approve", proposal["id"])
        approved = reopened.approve_pending_user_memory(
            proposal["id"], decision_event_id=command["id"])
        again = reopened.approve_pending_user_memory(proposal["id"])

        self.assertEqual(approved["status"], "approved")
        self.assertEqual(approved["decision_event_id"], command["id"])
        self.assertEqual(approved["resulting_memory_id"], again["resulting_memory_id"])
        record = reopened.get_by_id(approved["resulting_memory_id"])
        self.assertEqual(record["subject"], "user")
        self.assertEqual(record["source"], "user_approved_reflection")
        self.assertEqual(record["evidence_event_ids"], [event["id"]])
        self.assertEqual(len(reopened.recent_records()), 1)
        self.assertEqual(reopened.list_pending_user_memories(), [])

    def test_rejection_is_durable_and_never_forms_memory(self):
        proposal, _ = self.proposal()
        command = self.command_event("reject", proposal["id"])
        rejected = self.memories.reject_pending_user_memory(
            proposal["id"], decision_event_id=command["id"])
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(
            self.memories.reject_pending_user_memory(proposal["id"])["decided_at"],
            rejected["decided_at"],
        )
        reopened = memory.MemoryStore(self.memories.db_path)
        with self.assertRaises(ValueError):
            reopened.approve_pending_user_memory(proposal["id"])
        self.assertEqual(reopened.recent_records(), [])

    def test_approved_proposal_cannot_be_rejected(self):
        proposal, _ = self.proposal()
        command = self.command_event("approve", proposal["id"])
        self.memories.approve_pending_user_memory(
            proposal["id"], decision_event_id=command["id"])
        with self.assertRaises(ValueError):
            self.memories.reject_pending_user_memory(proposal["id"])

    def test_missing_or_invalid_historical_source_does_not_approve(self):
        proposal, event = self.proposal()
        command = self.command_event("approve", proposal["id"])
        with closing(sqlite3.connect(self.events.db_path)) as connection:
            with connection:
                connection.execute("UPDATE events SET actor = 'aperture' WHERE id = ?",
                                   (event["id"],))
        with self.assertRaisesRegex(ValueError, "sources"):
            self.memories.approve_pending_user_memory(
                proposal["id"], decision_event_id=command["id"])
        self.assertEqual(
            self.memories.get_pending_user_memory(proposal["id"])["status"],
            "pending",
        )
        self.assertEqual(self.memories.recent_records(), [])

    def test_supersession_preserves_history_and_rejects_stale_target(self):
        old = self.memories.remember(
            "The user prefers evening tea.", category="preference",
            subject="user",
        )
        proposal, event = self.proposal(supersedes=old["id"])
        command = self.command_event("approve", proposal["id"])
        decided = self.memories.approve_pending_user_memory(
            proposal["id"], decision_event_id=command["id"])
        previous = self.memories.get_by_id(old["id"])
        current = self.memories.get_by_id(decided["resulting_memory_id"])
        self.assertFalse(previous["active"])
        self.assertEqual(previous["valid_until"], current["valid_from"])
        self.assertEqual(current["supersedes_memory_id"], old["id"])
        self.assertEqual(current["evidence_event_ids"], [event["id"]])

        later, _ = self.proposal(supersedes=current["id"])
        later_command = self.command_event("approve", later["id"])
        self.memories.remember("The user prefers morning tea.", category="preference",
                               subject="user", importance=5)
        with self.assertRaisesRegex(ValueError, "target changed"):
            self.memories.approve_pending_user_memory(
                later["id"], decision_event_id=later_command["id"])
        self.assertEqual(self.memories.get_pending_user_memory(later["id"])["status"],
                         "pending")

    def test_insert_failure_rolls_back_target_and_decision(self):
        old = self.memories.remember("The user prefers evenings.", subject="user")
        proposal, _ = self.proposal(supersedes=old["id"])
        command = self.command_event("approve", proposal["id"])
        with patch.object(memory.uuid, "uuid4") as new_uuid:
            new_uuid.return_value.hex = old["id"] + "0" * 20
            with self.assertRaises(sqlite3.IntegrityError):
                self.memories.approve_pending_user_memory(
                    proposal["id"], decision_event_id=command["id"])
        self.assertTrue(self.memories.get_by_id(old["id"])["active"])
        self.assertIsNone(self.memories.get_by_id(old["id"])["valid_until"])
        self.assertEqual(self.memories.get_pending_user_memory(proposal["id"])["status"],
                         "pending")

    def test_exact_pending_proposal_is_deduplicated(self):
        event = self.user_event()
        values = dict(content="The user prefers morning tea.",
                      category="preference", importance=4,
                      evidence_event_ids=[event["id"]])
        first = self.memories.create_pending_user_memory(**values)
        second = self.memories.create_pending_user_memory(**values)
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(len(self.memories.list_pending_user_memories()), 1)

    def test_pending_decision_requires_observed_command_event(self):
        proposal, _ = self.proposal()
        for action in (self.memories.approve_pending_user_memory,
                       self.memories.reject_pending_user_memory):
            with self.assertRaisesRegex(ValueError, "requires an observed command"):
                action(proposal["id"])
        message = self.user_event("/memory approve " + proposal["id"])
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.memories.approve_pending_user_memory(
                proposal["id"], decision_event_id=message["id"])
        self.assertEqual(self.memories.get_pending_user_memory(proposal["id"])["status"],
                         "pending")
        self.assertEqual(self.memories.recent_records(), [])

    def test_untargeted_proposal_rejects_new_equivalent_current_memory(self):
        proposal, _ = self.proposal()
        current = self.memories.remember(
            "The user prefers morning tea.", category="preference",
            subject="user", evidence_event_ids=["old-evidence"],
        )
        command = self.command_event("approve", proposal["id"])
        with self.assertRaisesRegex(ValueError, "equivalent current user memory appeared"):
            self.memories.approve_pending_user_memory(
                proposal["id"], decision_event_id=command["id"])
        self.assertEqual(self.memories.get_pending_user_memory(proposal["id"])["status"],
                         "pending")
        self.assertEqual(self.memories.get_by_id(current["id"])["evidence_event_ids"],
                         ["old-evidence"])

    def test_decision_command_is_checked_and_audited(self):
        proposal, _ = self.proposal()
        wrong = self.command_event("reject", proposal["id"])
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.memories.approve_pending_user_memory(
                proposal["id"], decision_event_id=wrong["id"])
        self.assertEqual(self.memories.get_pending_user_memory(proposal["id"])["status"],
                         "pending")
        command = self.command_event("approve", proposal["id"])
        approved = self.memories.approve_pending_user_memory(
            proposal["id"], decision_event_id=command["id"])
        self.assertEqual(approved["decision_event_id"], command["id"])
        self.assertEqual(approved["status"], "approved")

    def test_existing_pending_schema_gains_decision_event_column(self):
        proposal, _ = self.proposal()
        with closing(sqlite3.connect(self.memories.db_path)) as connection:
            with connection:
                connection.execute(
                    "ALTER TABLE pending_user_memories DROP COLUMN decision_event_id"
                )
        reopened = memory.MemoryStore(self.memories.db_path)
        command = self.command_event("reject", proposal["id"])
        rejected = reopened.reject_pending_user_memory(
            proposal["id"], decision_event_id=command["id"])
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(rejected["decision_event_id"], command["id"])


if __name__ == "__main__":
    unittest.main()
