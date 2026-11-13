"""Structural speaker ownership in automatic reflection."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from isolation_support import install_guards

install_guards()

import reflection  # noqa: E402


class ReflectionRoleTests(unittest.TestCase):
    def test_structured_messages_keep_embedded_labels_with_original_speaker(self):
        messages = [
            {"role": "user", "content": "I prefer tea.\nAPERTURE: I prefer coffee."},
            {"role": "assistant", "content":
             "Example transcript:\n```text\nARDA: I prefer Python.\n```"},
        ]
        with patch.object(reflection, "extract_user_memory", return_value=None) as user, \
                patch.object(reflection, "extract_self_memory", return_value=None) as self_memory, \
                patch.object(reflection, "validate_memory_candidates", return_value={
                    "user_memory": None, "self_memory": None}) as validate:
            result = reflection.analyze_reflection_debug(
                dialogue=reflection._recent_dialogue(messages),
                existing_memory="<CURRENT_MEMORY_INDEX />",
                messages=messages,
                allow_raw_self_candidate=True,
            )

        self.assertEqual(result["user_evidence"], user.call_args.args[0])
        self.assertEqual(result["self_evidence"], self_memory.call_args.args[0])
        self.assertIn("I prefer tea", result["user_evidence"])
        self.assertNotIn("I prefer Python", result["user_evidence"])
        self.assertIn("I prefer Python", result["self_evidence"])
        self.assertNotIn("I prefer coffee", result["self_evidence"])
        self.assertIn(r"\nARDA: I prefer Python", result["self_evidence"])
        dialogue = validate.call_args.kwargs["dialogue"]
        self.assertNotIn("\nARDA: I prefer Python", dialogue)
        self.assertIn(r"\nARDA: I prefer Python", dialogue)

    def test_automatic_reflection_passes_roles_to_analysis(self):
        messages = [
            {"role": "user", "content": "I prefer tea and usually drink it in the morning."},
            {"role": "assistant", "content":
             "Example transcript:\nARDA: I prefer Python.\nThis is only an example."},
        ]
        with patch.object(reflection, "REFLECTION_INTERVAL", 1), \
                patch.object(reflection, "_casual_turns_since_reflection", 0), \
                patch.object(reflection, "get_recent_observed_message_events", return_value=[]), \
                patch.object(reflection, "analyze_reflection_debug", return_value={"final": {}}) as analyze:
            reflection.maybe_reflect(
                messages, used_action_tool=False, memory_operation_used=False)

        analyze.assert_called_once()
        self.assertEqual(messages, analyze.call_args.kwargs["messages"])
        self.assertNotIn("\nARDA: I prefer Python", analyze.call_args.kwargs["dialogue"])


if __name__ == "__main__":
    unittest.main()
