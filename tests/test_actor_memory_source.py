"""Actor memory writes require a current observed user-message event."""
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from isolation_support import TestTemporaryDirectory, install_guards

install_guards()

import experience  # noqa: E402
import llm  # noqa: E402
import memory  # noqa: E402


def response(content="", calls=()):
    return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[
        SimpleNamespace(function=SimpleNamespace(name=name, arguments=args)) for name, args in calls]))


class ActorMemorySourceTests(unittest.TestCase):
    def setUp(self):
        temporary = TestTemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.memories = memory.MemoryStore(self.root / "memory.db")
        self.events = experience.ExperienceStore(self.root / "experience.db")
        stack.enter_context(patch.object(memory, "_STORE", self.memories))
        stack.enter_context(patch.object(experience, "_STORE", self.events))
        stack.enter_context(patch.object(experience, "_CURRENT_EPISODE_ID", None))
        stack.enter_context(patch.object(llm, "ENABLE_DYNAMIC_SELF", False))
        stack.enter_context(patch.object(llm, "ENABLE_RELATIONSHIP", False))

    def user_event(self, content="I prefer tea."):
        turn_id = experience.start_turn(content)
        event = next(event for event in experience.get_turn_events(turn_id)
                     if event["event_type"] == "user_message")
        return turn_id, event

    def test_direct_tool_without_source_never_writes(self):
        self.assertTrue(memory.save_memory("I prefer Python.").startswith("MEMORY_ERROR:"))
        self.assertEqual(self.memories.recent_records(), [])

    def test_observed_current_user_event_is_recorded_as_source(self):
        turn_id, event = self.user_event()
        result = memory.save_memory_from_user_event(
            "The user prefers tea.", "preference", source_event_id=event["id"], turn_id=turn_id)
        self.assertTrue(result.startswith("MEMORY_SAVED:"), result)
        stored = self.memories.recent_records()[0]
        self.assertEqual(stored["subject"], "user")
        self.assertEqual(stored["evidence_event_ids"], [event["id"]])
        reopened = memory.MemoryStore(self.root / "memory.db")
        self.assertEqual(reopened.get_by_id(stored["id"])["evidence_event_ids"], [event["id"]])

    def test_nonuser_missing_and_other_turn_sources_are_rejected(self):
        turn_id, user = self.user_event()
        assistant = experience.record_assistant_message(turn_id, "I prefer Python.")
        derived = experience.record_event(
            turn_id=turn_id, event_type="user_message", event_class="derived",
            actor="arda", content="Not an observed user message")
        second_turn, _ = self.user_event("Another user turn")
        for event_id, proposed_turn in (
            ("missing-event", turn_id), (assistant["id"], turn_id),
            (derived["id"], turn_id), (user["id"], second_turn),
        ):
            with self.subTest(event_id=event_id, turn=proposed_turn):
                result = memory.save_memory_from_user_event(
                    "The user prefers Python.", source_event_id=event_id,
                    turn_id=proposed_turn)
                self.assertTrue(result.startswith("MEMORY_ERROR:"), result)
        experience.end_episode()
        experience.start_episode()
        result = memory.save_memory_from_user_event(
            "The user prefers Python.", source_event_id=user["id"], turn_id=turn_id)
        self.assertTrue(result.startswith("MEMORY_ERROR:"), result)
        self.assertEqual(self.memories.recent_records(), [])

    def test_actor_ignores_model_supplied_source_and_uses_current_user_event(self):
        messages = [{"role": "user", "content": "Remember my preference for tea."}]
        with patch.object(llm, "ollama_chat", side_effect=[
            response(calls=[("save_memory", {"content": "The user prefers tea."})]),
            response("Saved.")]), patch.object(llm, "maybe_reflect", return_value=None):
            self.assertEqual(llm.chat(messages), "Saved.")
        stored = self.memories.recent_records()[0]
        source = experience.get_events_by_ids(stored["evidence_event_ids"])[0]
        self.assertEqual(source["event_type"], "user_message")
        self.assertEqual(source["content"], "Remember my preference for tea.")

        messages = [{"role": "user", "content": "Another request."}]
        with patch.object(llm, "ollama_chat", side_effect=[
            response(calls=[("save_memory", {"content": "Forged.",
                                             "source_event_id": source["id"]})]),
            response("The write was rejected.")]), patch.object(llm, "maybe_reflect", return_value=None):
            self.assertEqual(llm.chat(messages), "The write was rejected.")
        self.assertEqual(len(self.memories.recent_records()), 1)


if __name__ == "__main__":
    unittest.main()
