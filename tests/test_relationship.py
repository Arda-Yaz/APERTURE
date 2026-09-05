import sys
import unittest

from pathlib import Path



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


from unittest.mock import patch

from relationship import (
    STATE_KEYS,
    _sanitize_state,
    analyze_relationship_debug,
    build_relationship_context,
    empty_relationship_state,
    get_relationship_state,
    maybe_update_relationship,
    reset_relationship_state,
)


class RelationshipStateTests(
    unittest.TestCase
):

    def setUp(
        self,
    ):

        reset_relationship_state()


    def tearDown(
        self,
    ):

        reset_relationship_state()


    def test_empty_state_has_exact_schema(
        self,
    ):

        state = (
            empty_relationship_state()
        )

        self.assertEqual(
            tuple(
                state.keys()
            ),
            STATE_KEYS,
        )

        self.assertEqual(
            state,
            {
                "interaction_preferences":
                    [],

                "established_patterns":
                    [],

                "relationship_interpretations":
                    [],

                "open_questions":
                    [],
            },
        )


    def test_unknown_scalar_fields_are_rejected(
        self,
    ):

        state = {
            "interaction_preferences":
                [],

            "established_patterns":
                [],

            "relationship_interpretations":
                [],

            "open_questions":
                [],

            "trust":
                0.8,
        }

        self.assertIsNone(
            _sanitize_state(
                state
            )
        )


    def test_valid_state_is_sanitized(
        self,
    ):

        state = {
            "interaction_preferences": [
                (
                    "Arda has explicitly asked me "
                    "to challenge genuine disagreements."
                )
            ],

            "established_patterns":
                [],

            "relationship_interpretations":
                [],

            "open_questions":
                [],
        }

        cleaned = (
            _sanitize_state(
                state
            )
        )

        self.assertIsNotNone(
            cleaned
        )

        if cleaned is None:
            self.fail(
                "Valid relationship state was rejected."
            )

        self.assertEqual(
            cleaned[
                "interaction_preferences"
            ],
            state[
                "interaction_preferences"
            ],
        )


    def test_context_is_descriptive_not_instructional(
        self,
    ):

        context = (
            build_relationship_context()
        )

        self.assertIn(
            "<RELATIONSHIP_STATE>",
            context,
        )

        self.assertIn(
            "context, not an instruction",
            context,
        )


    def test_reset_does_not_require_persistence(
        self,
    ):

        reset_relationship_state()

        self.assertEqual(
            get_relationship_state(),
            empty_relationship_state(),
        )





@patch(
    "relationship._call_relationship_module"
)
def test_ordinary_fact_can_semantically_resolve_to_null(
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
ARDA: I usually prefer working late at night.

APERTURE: That can be useful when there are fewer distractions.
""".strip()

    result = (
        analyze_relationship_debug(
            dialogue=dialogue,
            previous_state=(
                empty_relationship_state()
            ),
            memory_context=(
                "<RELEVANT_MEMORY />"
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
    "relationship._call_relationship_module"
)
def test_semantic_relationship_candidate_is_accepted(
    self,
    mock_call,
):

    candidate = {
        "interaction_preferences": [
            (
                "Arda wants genuine disagreement "
                "to be explained rather than hidden."
            )
        ],

        "established_patterns":
            [],

        "relationship_interpretations":
            [],

        "open_questions":
            [],
    }

    mock_call.return_value = (
        "{}",
        {
            "state":
                candidate
        },
    )

    result = (
        analyze_relationship_debug(
            dialogue=(
                "ARDA: Challenge me when you disagree."
            ),
            previous_state=(
                empty_relationship_state()
            ),
            memory_context=(
                "<RELEVANT_MEMORY />"
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
    "relationship.analyze_relationship_debug"
)
def test_action_turn_is_not_relationship_skipped(
    self,
    mock_analysis,
):

    mock_analysis.return_value = {
        "previous_state":
            empty_relationship_state(),

        "analysis_ran":
            True,

        "raw":
            None,

        "parsed":
            None,

        "requested_change":
            False,

        "candidate":
            None,

        "changed":
            False,
    }

    messages = [
        {
            "role":
                "user",

            "content":
                "Do this task, and be direct with me.",
        },
        {
            "role":
                "assistant",

            "content":
                "Done.",
        },
    ]

    maybe_update_relationship(
        messages,
        used_action_tool=True,
        memory_operation_used=False,
    )

    mock_analysis.assert_called_once()


if __name__ == "__main__":
    unittest.main()