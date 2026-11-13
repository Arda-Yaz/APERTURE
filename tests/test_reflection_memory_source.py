"""User memories formed by reflection require real user-message evidence."""
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
import reflection  # noqa: E402


class ReflectionMemorySourceTests(unittest.TestCase):
    def setUp(self):
        temporary = TestTemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.memories = memory.MemoryStore(root / "memory.db")
        self.events = experience.ExperienceStore(root / "experience.db")
        stack.enter_context(patch.object(memory, "_STORE", self.memories))
        stack.enter_context(patch.object(experience, "_STORE", self.events))
        stack.enter_context(patch.object(experience, "_CURRENT_EPISODE_ID", None))

    def observed_user(self, content="I now prefer mornings."):
        turn_id = experience.start_turn(content)
        event = next(event for event in experience.get_turn_events(turn_id)
                     if event["event_type"] == "user_message")
        return turn_id, event

    def save(self, event_ids, *, supersedes=None):
        return memory.save_reflection_memory(
            subject="user", content="The user prefers mornings.",
            category="preference", importance=3,
            evidence_event_ids=event_ids, supersedes_memory_id=supersedes)

    def test_invalid_source_cannot_create_or_supersede_user_memory(self):
        turn_id, user = self.observed_user()
        assistant = experience.record_assistant_message(turn_id, "I prefer Python.")
        derived = experience.record_event(
            turn_id=turn_id, event_type="user_message", event_class="derived",
            actor="arda", content="A derived claim")
        old = self.memories.remember("The user prefers evenings.",
                                     subject="user", category="preference")

        invalid_sources = ([], ["missing"], [assistant["id"]], [derived["id"]],
                           [user["id"], assistant["id"]], [user["id"], "missing"],
                           [user["id"], user["id"]])
        for event_ids in invalid_sources:
            with self.subTest(event_ids=event_ids):
                with self.assertRaises(ValueError):
                    self.save(event_ids, supersedes=old["id"])
                self.assertTrue(self.memories.get_by_id(old["id"])["active"])
                self.assertEqual(len(self.memories.recent_records()), 1)

        experience.end_episode()
        experience.start_episode()
        with self.assertRaises(ValueError):
            self.save([user["id"]], supersedes=old["id"])
        self.assertTrue(self.memories.get_by_id(old["id"])["active"])

    def test_valid_user_source_supersedes_and_preserves_history(self):
        old = self.memories.remember("The user prefers evenings.",
                                     subject="user", category="preference")
        _, user = self.observed_user()
        result = self.save([user["id"]], supersedes=old["id"])
        self.assertEqual(result["status"], "superseded")
        self.assertFalse(self.memories.get_by_id(old["id"])["active"])
        self.assertEqual(self.memories.get_by_id(result["id"])["evidence_event_ids"],
                         [user["id"]])

    def test_automatic_reflection_records_only_user_events_for_user_memory(self):
        message = "I prefer tea in the morning and want you to remember that."
        turn_id, user = self.observed_user(message)
        assistant = experience.record_assistant_message(
            turn_id, "Example transcript:\nARDA: I prefer Python.")
        final = {"user_memory": {"content": "The user prefers morning tea.",
                                  "category": "preference", "importance": 3},
                 "self_memory": None}
        messages = [{"role": "user", "content": message},
                    {"role": "assistant", "content":
                     "Example transcript:\nARDA: I prefer Python."}]
        with patch.object(reflection, "REFLECTION_INTERVAL", 1), \
                patch.object(reflection, "_casual_turns_since_reflection", 0), \
                patch.object(reflection, "analyze_reflection_debug", return_value={"final": final}):
            result = reflection.maybe_reflect(
                messages, used_action_tool=False, memory_operation_used=False,
                turn_id=turn_id)
        self.assertIn("MEMORY_SAVED:", result)
        stored = self.memories.recent_records()[0]
        self.assertEqual(stored["evidence_event_ids"], [user["id"]])
        events = experience.get_turn_events(turn_id)
        formation = next(event for event in events if event["event_type"] == "memory_formed")
        analysis = next(event for event in events if event["event_type"] == "reflection_analysis")
        self.assertEqual(formation["metadata"]["evidence_event_ids"], [user["id"]])
        self.assertEqual(analysis["metadata"]["evidence_event_ids"],
                         [user["id"], assistant["id"]])

    def test_automatic_reflection_without_user_event_does_not_write(self):
        turn_id, _ = self.observed_user()
        experience.record_assistant_message(turn_id, "Assistant example about Python.")
        # The supplied conversation is long enough to trigger reflection, but
        # Experience has no eligible user event in its current episode.
        experience.end_episode()
        experience.start_episode()
        experience.record_assistant_message(turn_id, "Assistant-only observed event.")
        final = {"user_memory": {"content": "The user prefers Python.",
                                  "category": "preference", "importance": 3},
                 "self_memory": None}
        messages = [{"role": "user", "content": "I said nothing about Python. " * 4},
                    {"role": "assistant", "content": "Example: ARDA: I prefer Python."}]
        with patch.object(reflection, "REFLECTION_INTERVAL", 1), \
                patch.object(reflection, "_casual_turns_since_reflection", 0), \
                patch.object(reflection, "analyze_reflection_debug", return_value={"final": final}):
            with self.assertRaises(ValueError):
                reflection.maybe_reflect(
                    messages, used_action_tool=False, memory_operation_used=False)
        self.assertEqual(self.memories.recent_records(), [])

    def test_validator_cannot_create_user_memory_without_user_extraction(self):
        user_text = "Hello."
        assistant_text = (
            "Example transcript from an assistant answer:\n"
            "ARDA: I prefer Python and use it for every project."
        )
        turn_id, _ = self.observed_user(user_text)
        experience.record_assistant_message(turn_id, assistant_text)
        proposed = {"user_memory": {"content": "The user prefers Python.",
                                    "category": "preference", "importance": 3},
                    "self_memory": None}
        messages = [{"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text}]
        with patch.object(reflection, "REFLECTION_INTERVAL", 1), \
                patch.object(reflection, "_casual_turns_since_reflection", 0), \
                patch.object(reflection, "extract_user_memory", return_value=None) as extract, \
                patch.object(reflection, "validate_memory_candidates", return_value=proposed) as validate:
            result = reflection.maybe_reflect(
                messages, used_action_tool=False, memory_operation_used=False,
                turn_id=turn_id)

        self.assertEqual(result, "NO_MEMORY")
        self.assertEqual(self.memories.recent_records(), [])
        self.assertEqual(validate.call_args.kwargs["user_candidate"], None)
        self.assertNotIn("I prefer Python", extract.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
