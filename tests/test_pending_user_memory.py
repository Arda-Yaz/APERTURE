"""Pending user memories preserve evidence without changing active memory."""

import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from isolation_support import TestTemporaryDirectory, install_guards

install_guards()

import experience  # noqa: E402
import memory  # noqa: E402


class PendingUserMemoryTests(unittest.TestCase):
    def setUp(self):
        temporary = TestTemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.db_path = self.root / "memory.db"
        self.memories = memory.MemoryStore(self.db_path)
        self.events = experience.ExperienceStore(self.root / "experience.db")
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(experience, "_STORE", self.events))
        stack.enter_context(patch.object(experience, "_CURRENT_EPISODE_ID", None))

    def user_event(self, content="I prefer quiet mornings."):
        turn_id = experience.start_turn(content)
        return turn_id, next(event for event in experience.get_turn_events(turn_id)
                             if event["event_type"] == "user_message")

    def queue(self, event_ids, **kwargs):
        return self.memories.create_pending_user_memory(
            content="The user prefers quiet mornings.", category="preference",
            importance=3, evidence_event_ids=event_ids, **kwargs)

    def test_pending_candidate_survives_reopen_without_forming_memory(self):
        _, first = self.user_event()
        _, second = self.user_event("I can work in the morning.")
        proposal = self.queue([second["id"], first["id"]])
        self.assertEqual(proposal["status"], "pending")
        self.assertEqual(proposal["evidence_event_ids"], [second["id"], first["id"]])
        self.assertEqual(proposal["source_episode_id"], first["episode_id"])
        self.assertEqual(proposal["content"], "The user prefers quiet mornings.")
        self.assertEqual(proposal["category"], "preference")
        self.assertEqual(proposal["importance"], 3)
        self.assertIsNone(proposal["decided_at"])
        self.assertIsNone(proposal["resulting_memory_id"])
        self.assertEqual(self.memories.recent_records(), [])

        reopened = memory.MemoryStore(self.db_path)
        self.assertEqual(reopened.get_pending_user_memory(proposal["id"]), proposal)
        self.assertEqual(reopened.list_pending_user_memories(), [proposal])
        self.assertIsNone(reopened.get_pending_user_memory("missing"))
        self.assertEqual(reopened.recent_records(), [])

    def test_invalid_or_nonuser_evidence_cannot_queue(self):
        turn_id, user = self.user_event()
        assistant = experience.record_assistant_message(turn_id, "I prefer Python.")
        derived = experience.record_event(
            turn_id=turn_id, event_class="derived", event_type="user_message",
            actor="arda", content="Inferred user claim")
        invalid = ([], ["unknown"], [assistant["id"]], [derived["id"]],
                   [user["id"], "unknown"], [user["id"], assistant["id"]],
                   [user["id"], user["id"]], [user["id"], " " + user["id"]],
                   [""], [None], "not-a-list")
        for ids in invalid:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.queue(ids)
        self.assertEqual(self.memories.list_pending_user_memories(), [])
        self.assertEqual(self.memories.recent_records(), [])

    def test_source_must_belong_to_current_episode(self):
        _, user = self.user_event()
        experience.end_episode()
        experience.start_episode()
        with self.assertRaises(ValueError):
            self.queue([user["id"]])
        self.assertEqual(self.memories.list_pending_user_memories(), [])

    def test_target_is_snapshotted_only_when_current_and_user_owned(self):
        _, user = self.user_event()
        old = self.memories.remember("The user prefers evenings.", subject="user")
        other = self.memories.remember("I prefer evenings.", subject="aperture")
        proposal = self.queue([user["id"]], supersedes_memory_id=old["id"])
        self.assertEqual(proposal["supersedes_memory_id"], old["id"])
        self.assertEqual(proposal["target_content"], old["content"])
        self.assertEqual(proposal["target_updated_at"],
                         self.memories.get_by_id(old["id"])["updated_at"])
        self.assertTrue(self.memories.get_by_id(old["id"])["active"])
        for target in ("missing", other["id"], ""):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.queue([user["id"]], supersedes_memory_id=target)
        self.memories.forget(old["id"])
        with self.assertRaises(ValueError):
            self.queue([user["id"]], supersedes_memory_id=old["id"])
        self.assertEqual(len(self.memories.list_pending_user_memories()), 1)


if __name__ == "__main__":
    unittest.main()
