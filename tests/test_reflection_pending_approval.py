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
import main  # noqa: E402
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
        stack.enter_context(patch.object(reflection, "REQUIRE_REFLECTION_USER_MEMORY_APPROVAL", False))

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

    def test_policy_comparison_with_faithful_and_incorrect_proposals(self):
        cases = (
            ("On weekday mornings I prefer brewing tea before starting work, because it helps me ease into the day.",
             "Arda prefers to brew tea before starting work on weekday mornings.", True),
            ("I still prefer evening walks, though yesterday I took one before breakfast.",
             "Arda now prefers morning walks.", False),
        )
        for require_approval, expected in ((False, (1, 1, 0)), (True, (0, 1, 2))):
            with (
                self.subTest(require_approval=require_approval),
                TestTemporaryDirectory() as directory,
                ExitStack() as stack,
            ):
                root = Path(directory)
                store = memory.MemoryStore(root / "memory.db")
                stack.enter_context(patch.object(memory, "_STORE", store))
                stack.enter_context(patch.object(experience, "_STORE", experience.ExperienceStore(root / "exp.db")))
                stack.enter_context(patch.object(experience, "_CURRENT_EPISODE_ID", None))
                stack.enter_context(patch.object(reflection, "REFLECTION_INTERVAL", 1))
                stack.enter_context(patch.object(reflection, "_casual_turns_since_reflection", 0))
                stack.enter_context(patch.object(reflection, "REQUIRE_REFLECTION_USER_MEMORY_APPROVAL", require_approval))
                previous = store.remember("Arda prefers evening walks.", category="preference", subject="user")
                incorrect_activations = faithful_retained = review_requests = 0
                for message, content, faithful in cases:
                    turn = experience.start_turn(message)
                    candidate = {"content": content, "category": "preference",
                                 "importance": 3, "needs_confirmation": False}
                    if not faithful:
                        candidate["supersedes_memory_id"] = previous["id"]
                    final = {"user_memory": candidate, "self_memory": None}
                    with patch.object(reflection, "analyze_reflection_debug", return_value={"final": final}):
                        outcome = reflection.maybe_reflect(
                            [{"role": "user", "content": message},
                             {"role": "assistant", "content": "I understand what you mean."}],
                            used_action_tool=False, memory_operation_used=False, turn_id=turn)
                    pending = store.list_pending_user_memories()
                    review_requests += len(pending)
                    if pending:
                        proposal, = pending
                        self.assertIn("MEMORY_PENDING", outcome)
                        self.assertTrue(store.get_by_id(previous["id"])["active"])
                        self.assertEqual(proposal["supersedes_memory_id"], candidate.get("supersedes_memory_id"))
                        decision = "approve" if faithful else "reject"
                        main.handle_memory_command(f"/memory {decision} {proposal['id']}")
                        if not faithful:
                            self.assertTrue(store.get_by_id(previous["id"])["active"])
                    active = any(item["content"] == content and item["active"]
                                 for item in store.recent_records())
                    faithful_retained += faithful and active
                    incorrect_activations += not faithful and active
                self.assertEqual((incorrect_activations, faithful_retained, review_requests), expected)

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
