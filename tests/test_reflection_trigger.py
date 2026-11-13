"""Automatic reflection runs only after the configured number of casual turns."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from isolation_support import install_guards

install_guards()

import reflection  # noqa: E402


class ReflectionTriggerTests(unittest.TestCase):
    def test_action_and_memory_turns_do_not_trigger_or_advance_reflection(self):
        messages = [
            {"role": "user", "content": "I prefer tea in the morning and work from home most weekdays."},
            {"role": "assistant", "content": "I will keep the distinction between your preferences and mine."},
        ]

        for used_action_tool, memory_operation_used in ((True, False), (False, True), (True, True)):
            with self.subTest(action=used_action_tool, memory=memory_operation_used):
                with patch.object(reflection, "REFLECTION_INTERVAL", 2), \
                        patch.object(reflection, "_casual_turns_since_reflection", 1), \
                        patch.object(reflection, "_recent_dialogue", wraps=reflection._recent_dialogue) as dialogue, \
                        patch.object(reflection, "get_recent_observed_message_events", return_value=[]) as events, \
                        patch.object(reflection, "analyze_reflection_debug", return_value={"final": {}}) as analyze:
                    result = reflection.maybe_reflect(
                        messages, used_action_tool=used_action_tool,
                        memory_operation_used=memory_operation_used)
                    self.assertIsNone(result)
                    self.assertEqual(reflection._casual_turns_since_reflection, 1)
                    dialogue.assert_not_called()
                    events.assert_not_called()
                    analyze.assert_not_called()

                    result = reflection.maybe_reflect(
                        messages, used_action_tool=False, memory_operation_used=False)
                    self.assertEqual(result, "NO_MEMORY")
                    self.assertEqual(reflection._casual_turns_since_reflection, 0)
                    analyze.assert_called_once()

    def test_skipped_turn_does_not_count_toward_next_interval(self):
        messages = [{"role": "user", "content": "A synthetic conversation with enough text to be considered for reflection."}]
        with patch.object(reflection, "REFLECTION_INTERVAL", 2), \
                patch.object(reflection, "_casual_turns_since_reflection", 0), \
                patch.object(reflection, "analyze_reflection_debug") as analyze:
            self.assertIsNone(reflection.maybe_reflect(
                messages, used_action_tool=True, memory_operation_used=False))
            self.assertIsNone(reflection.maybe_reflect(
                messages, used_action_tool=False, memory_operation_used=False))
            self.assertEqual(reflection._casual_turns_since_reflection, 1)
            analyze.assert_not_called()


if __name__ == "__main__":
    unittest.main()
