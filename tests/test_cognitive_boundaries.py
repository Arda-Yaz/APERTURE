import sys
import unittest

from pathlib import Path
from unittest.mock import patch


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

APP_DIR = (
    ROOT
    / "app"
)

sys.path.insert(
    0,
    str(APP_DIR),
)


from self_state import (  # noqa: E402
    STATE_KEYS,
    _sanitize_state,
    analyze_self_state_debug,
    empty_self_state,
    maybe_update_self_state,
    reset_self_state,
    _current_turn_dialogue,
)

from reflection import (  # noqa: E402
    SELF_MEMORY_CATEGORIES,
    analyze_reflection_debug,
    _split_dialogue_by_speaker,
     analyze_self_consolidation,
)



class DynamicSelfBoundaryTests(
    unittest.TestCase
):

    def setUp(
        self,
    ):

        reset_self_state()


    def tearDown(
        self,
    ):

        reset_self_state()


    @patch(
        "self_state._call_self_state_module"
    )
    def test_ordinary_language_is_semantically_analyzed(
        self,
        mock_call,
    ):

        mock_call.return_value = (
            '{"state": null}',
            {
                "state":
                    None
            },
        )

        dialogue = """
ARDA: I usually work late at night.

APERTURE: That's interesting. How does that affect your schedule?
""".strip()

        result = (
            analyze_self_state_debug(
                dialogue=dialogue,
                previous_state=(
                    empty_self_state()
                ),
            )
        )

        mock_call.assert_called_once()

        self.assertTrue(
            result[
                "analysis_ran"
            ]
        )

        self.assertIsNone(
            result[
                "candidate"
            ]
        )


    @patch(
        "self_state._call_self_state_module"
    )
    def test_semantic_result_does_not_require_trigger_phrase(
        self,
        mock_call,
    ):

        candidate = {
            "current_orientation":
                [],

            "active_interests": [
                (
                    "I find the question of persistent "
                    "identity worth exploring further."
                )
            ],

            "unresolved_positions":
                [],
        }

        mock_call.return_value = (
            "{}",
            {
                "state":
                    candidate
            },
        )

        dialogue = """
ARDA: What has been on your mind?

APERTURE: The question of persistent identity keeps drawing my attention in a way I would like to explore further.
""".strip()

        result = (
            analyze_self_state_debug(
                dialogue=dialogue,
                previous_state=(
                    empty_self_state()
                ),
            )
        )

        self.assertEqual(
            result[
                "candidate"
            ],
            candidate,
        )


    @patch(
        "reflection._call_memory_module"
    )
    def test_self_consolidation_uses_validated_state(
        self,
        mock_call,
    ):

        mock_call.return_value = {
            "candidate": {
                "content":
                    "I currently lean toward Python.",

                "category":
                    "preference",

                "importance":
                    3,

                "supersedes_memory_id":
                    None,
            }
        }

        result = (
            analyze_self_consolidation(
                state_update={
                    "current_orientation": [
                        (
                            "I currently lean "
                            "toward Python."
                        )
                    ],
                    "active_interests":
                        [],
                    "unresolved_positions":
                        [],
                },
                assistant_evidence=(
                    "Right now I lean toward Python."
                ),
                existing_memory=(
                    "<CURRENT_MEMORY_INDEX />"
                ),
            )
        )

        self.assertIsNotNone(
            result
        )

        if result is None:
            self.fail(
                "Expected a sanitized result."
            )

        self.assertEqual(
            result[
                "category"
            ],
            "preference",
        )


    def test_unknown_dynamic_self_fields_are_rejected(
        self,
    ):

        state = {
            "current_orientation":
                [],

            "active_interests":
                [],

            "unresolved_positions":
                [],

            "personality_score":
                0.8,
        }

        self.assertIsNone(
            _sanitize_state(
                state
            )
        )


    @patch(
        "self_state.analyze_self_state"
    )


    def test_action_turn_is_not_semantically_skipped(
        self,
        mock_analysis,
    ):

        mock_analysis.return_value = (
            None
        )

        messages = [
            {
                "role":
                    "user",

                "content":
                    "Read this file and tell me what you think.",
            },
            {
                "role":
                    "assistant",

                "content":
                    "I finished reading it.",
            },
        ]

        maybe_update_self_state(
            messages,
            used_action_tool=True,
            memory_operation_used=False,
        )

        mock_analysis.assert_called_once()


@patch(
    "reflection.extract_self_memory"
)
def test_automatic_reflection_does_not_use_raw_actor_for_self_memory(
    self,
    mock_self,
):

    dialogue = """
ARDA: I work late at night.

APERTURE: If you need help during that time, I am here to assist.
""".strip()

    result = (
        analyze_reflection_debug(
            dialogue=dialogue,
            existing_memory=(
                "<CURRENT_MEMORY_INDEX />"
            ),
        )
    )

    mock_self.assert_not_called()

    self.assertIsNone(
        result[
            "self_candidate"
        ]
    )

    self.assertIsNone(
        result[
            "final"
        ][
            "self_memory"
        ]
    )

                                            
    def test_reflection_self_evidence_uses_latest_aperture_turn_only(   
        self,
    ):

        dialogue = """
    ARDA: Challenge me when you disagree.

    APERTURE: I aim to engage in meaningful critical dialogue.

    ARDA: I prefer working late.

    APERTURE: That's useful to know.

    ARDA: I play games before working.

    APERTURE: Balancing both can be difficult.
    """.strip()

        (
            user_evidence,
            self_evidence,
        ) = (
            _split_dialogue_by_speaker(
                dialogue
            )
        )

        self.assertIn(
            "I play games before working",
            user_evidence,
        )

        self.assertIn(
            "Balancing both can be difficult",
            self_evidence,
        )

        self.assertNotIn(
            "I aim to engage",
            self_evidence,
        )


    def test_dynamic_self_current_turn_excludes_old_assistant_evidence(
    self,
    ):

        messages = [
            {
                "role":
                    "user",

                "content":
                    "Challenge me when you disagree.",
            },
            {
                "role":
                    "assistant",

                "content":
                    (
                        "I aim to engage in meaningful "
                        "critical dialogue."
                    ),
            },
            {
                "role":
                    "user",

                "content":
                    "I prefer working late.",
            },
            {
                "role":
                    "assistant",

                "content":
                    (
                        "If you need help during late-night "
                        "work, let me know."
                    ),
            },
        ]

        dialogue = (
            _current_turn_dialogue(
                messages
            )
        )

        self.assertNotIn(
            "I aim to engage",
            dialogue,
        )

        self.assertIn(
            "I prefer working late",
            dialogue,
        )

        self.assertIn(
            "If you need help",
            dialogue,
        )


class ReflectionBoundaryTests(
    unittest.TestCase
):

    def test_relationship_has_single_cognitive_owner(
        self,
    ):

        self.assertNotIn(
            "relationship",
            SELF_MEMORY_CATEGORIES,
        )


if __name__ == "__main__":
    unittest.main()