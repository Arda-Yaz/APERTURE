"""Reflection queues uncertain user memories without granting model approval."""
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


class ReflectionPendingApprovalTests(unittest.TestCase):
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

    def test_validation_requires_structured_confirmation_and_user_extraction(self):
        candidate = {"content": "Arda prefers quiet mornings.",
                     "category": "preference", "importance": 3}
        for value, expected in ((True, True), (False, False), ("yes", None)):
            with self.subTest(value=value):
                response = {"user_memory": dict(candidate, needs_confirmation=value),
                            "self_memory": None}
                with patch.object(reflection, "_call_memory_module", return_value=response):
                    result = reflection.validate_memory_candidates(
                        "ARDA: I prefer quiet mornings.", candidate, None, "")
                if expected is None:
                    self.assertIsNone(result["user_memory"])
                else:
                    self.assertIs(result["user_memory"]["needs_confirmation"], expected)

        with patch.object(reflection, "_call_memory_module", return_value={
            "user_memory": candidate, "self_memory": None}):
            result = reflection.validate_memory_candidates(
                "ARDA: I prefer quiet mornings.", candidate, None, "")
        self.assertIsNone(result["user_memory"])

        with patch.object(reflection, "_call_memory_module", return_value={
            "user_memory": dict(candidate, needs_confirmation=True), "self_memory": None}):
            result = reflection.validate_memory_candidates(
                "ARDA: Hello.", None, None, "")
        self.assertIsNone(result["user_memory"])

    def test_uncertain_candidate_is_pending_and_does_not_form_memory(self):
        message = "I may prefer working on quiet mornings for long projects."
        turn_id = experience.start_turn(message)
        source = next(event for event in experience.get_turn_events(turn_id)
                      if event["event_type"] == "user_message")
        final = {"user_memory": {"content": "Arda may prefer quiet mornings.",
                                 "category": "preference", "importance": 3,
                                 "needs_confirmation": True},
                 "self_memory": None}
        messages = [{"role": "user", "content": message},
                    {"role": "assistant", "content": "That sounds possible."}]
        with patch.object(reflection, "REFLECTION_INTERVAL", 1), \
                patch.object(reflection, "_casual_turns_since_reflection", 0), \
                patch.object(reflection, "analyze_reflection_debug", return_value={"final": final}):
            result = reflection.maybe_reflect(
                messages, used_action_tool=False, memory_operation_used=False,
                turn_id=turn_id)

        pending = self.memories.list_pending_user_memories()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["evidence_event_ids"], [source["id"]])
        self.assertEqual(self.memories.recent_records(), [])
        self.assertIn(f"/memory approve {pending[0]['id']}", result)
        self.assertIn("MEMORY_PENDING", result)
        self.assertNotIn("MEMORY_SAVED", result)
        self.assertFalse(any(event["event_type"] == "memory_formed"
                             for event in experience.get_turn_events(turn_id)))

    def test_invalid_confirmation_flag_is_not_automatic_approval(self):
        message = "I prefer quiet mornings for long projects."
        turn_id = experience.start_turn(message)
        final = {"user_memory": {"content": "Arda prefers quiet mornings.",
                                 "category": "preference", "importance": 3,
                                 "needs_confirmation": "false"},
                 "self_memory": None}
        with patch.object(reflection, "REFLECTION_INTERVAL", 1), \
                patch.object(reflection, "_casual_turns_since_reflection", 0), \
                patch.object(reflection, "analyze_reflection_debug", return_value={"final": final}):
            result = reflection.maybe_reflect(
                [{"role": "user", "content": message},
                 {"role": "assistant", "content":
                  "Understood. We can discuss whether that preference will last."}],
                used_action_tool=False, memory_operation_used=False,
                turn_id=turn_id)
        self.assertEqual(result, "NO_MEMORY")
        self.assertEqual(self.memories.recent_records(), [])
        self.assertEqual(self.memories.list_pending_user_memories(), [])

    def test_missing_confirmation_flag_cannot_write_memory(self):
        message = "I prefer quiet mornings for long projects."
        turn_id = experience.start_turn(message)
        final = {"user_memory": {"content": "Arda prefers quiet mornings.",
                                 "category": "preference", "importance": 3},
                 "self_memory": None}
        with patch.object(reflection, "REFLECTION_INTERVAL", 1), \
                patch.object(reflection, "_casual_turns_since_reflection", 0), \
                patch.object(reflection, "analyze_reflection_debug", return_value={"final": final}):
            result = reflection.maybe_reflect(
                [{"role": "user", "content": message}], used_action_tool=False,
                memory_operation_used=False, turn_id=turn_id)
        self.assertIsNone(result)
        self.assertEqual(self.memories.recent_records(), [])


if __name__ == "__main__":
    unittest.main()
