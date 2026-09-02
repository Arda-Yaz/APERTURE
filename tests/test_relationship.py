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


from relationship import (  # noqa: E402
    STATE_KEYS,
    _has_relationship_signal,
    _sanitize_state,
    build_relationship_context,
    empty_relationship_state,
    get_relationship_state,
    reset_relationship_state,
    _enforce_evidence_boundaries,
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

    def test_compliance_does_not_create_relationship_interpretation(
        self,
    ):

        candidate = {
            "interaction_preferences": [
                (
                    "Arda asked APERTURE to "
                    "challenge genuine disagreements."
                )
            ],
            "established_patterns":
                [],
            "relationship_interpretations": [
                (
                    "APERTURE interprets this as "
                    "a preference for thoughtful dialogue."
                )
            ],
            "open_questions":
                [],
        }

        dialogue = """
    ARDA: When you disagree with me, challenge my reasoning.

    APERTURE: Understood. I will challenge your reasoning when I genuinely disagree.
    """.strip()

        cleaned = (
            _enforce_evidence_boundaries(
                candidate,
                dialogue,
            )
        )
        assert cleaned is not None

        self.assertEqual(
            cleaned[
                "interaction_preferences"
            ],
            candidate[
                "interaction_preferences"
            ],
        )

        self.assertEqual(
            cleaned[
                "relationship_interpretations"
            ],
            [],
        )


    def test_explicit_relationship_interpretation_is_preserved(
        self,
    ):

        candidate = {
            "interaction_preferences":
                [],
            "established_patterns":
                [],
            "relationship_interpretations": [
                (
                    "I currently see our discussions "
                    "as increasingly exploratory."
                )
            ],
            "open_questions":
                [],
        }

        dialogue = """
    APERTURE: I think our discussions have become increasingly exploratory.
    """.strip()

        cleaned = (
            _enforce_evidence_boundaries(
                candidate,
                dialogue,
            )
        )
        assert cleaned is not None

        self.assertEqual(
            cleaned[
                "relationship_interpretations"
            ],
            candidate[
                "relationship_interpretations"
            ],
        )


    

    def test_user_fact_is_not_relationship_signal(
        self,
    ):

        dialogue = """
ARDA: I usually prefer working late at night.

APERTURE: That can be useful when there are fewer distractions.
""".strip()

        self.assertFalse(
            _has_relationship_signal(
                dialogue
            )
        )


    def test_explicit_interaction_preference_is_signal(
        self,
    ):

        dialogue = """
ARDA: When you disagree with me, challenge me instead of just agreeing.

APERTURE: Understood.
""".strip()

        self.assertTrue(
            _has_relationship_signal(
                dialogue
            )
        )


    def test_relationship_interpretation_is_signal(
        self,
    ):

        dialogue = """
APERTURE: I think our conversations are becoming more exploratory.
""".strip()

        self.assertTrue(
            _has_relationship_signal(
                dialogue
            )
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


if __name__ == "__main__":
    unittest.main()