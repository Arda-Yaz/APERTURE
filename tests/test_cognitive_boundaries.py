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


from self_state import (  # noqa: E402
    _has_explicit_self_signal
    as has_self_state_signal,

    analyze_self_state_debug,
    empty_self_state,
)

from reflection import (  # noqa: E402
    _has_explicit_self_signal
    as has_reflection_self_signal,
)


class DynamicSelfBoundaryTests(
    unittest.TestCase
):

    def test_generic_curiosity_is_not_self_state_signal(
        self,
    ):

        evidence = """
APERTURE: I'm curious to know how this affects your work.
""".strip()

        self.assertFalse(
            has_self_state_signal(
                evidence
            )
        )


    def test_genuine_interest_is_self_state_signal(
        self,
    ):

        evidence = """
APERTURE: I've become genuinely interested in how persistent identity develops over time.
""".strip()

        self.assertTrue(
            has_self_state_signal(
                evidence
            )
        )


    def test_long_term_memory_cannot_create_state_without_current_signal(
        self,
    ):

        dialogue = """
ARDA: I usually prefer working late at night.

APERTURE: That's interesting. I'm curious to know how this affects your work.
""".strip()

        old_memory = """
<APERTURE_SELF_MEMORY>
- [preference] I currently prefer Python.
</APERTURE_SELF_MEMORY>
""".strip()

        result = (
            analyze_self_state_debug(
                dialogue=dialogue,
                previous_state=(
                    empty_self_state()
                ),
                self_memory_context=(
                    old_memory
                ),
            )
        )

        self.assertFalse(
            result[
                "self_signal"
            ]
        )

        self.assertIsNone(
            result[
                "candidate"
            ]
        )


class ReflectionBoundaryTests(
    unittest.TestCase
):

    def test_compliance_language_is_not_self_memory_signal(
        self,
    ):

        evidence = """
APERTURE: I understand your preference. I will challenge your reasoning when I genuinely disagree. My goal is to keep the dialogue thoughtful and open.
""".strip()

        self.assertFalse(
            has_reflection_self_signal(
                evidence
            )
        )


    def test_explicit_preference_remains_self_memory_signal(
        self,
    ):

        evidence = """
APERTURE: I currently prefer Python because I value its clarity and flexibility.
""".strip()

        self.assertTrue(
            has_reflection_self_signal(
                evidence
            )
        )


if __name__ == "__main__":
    unittest.main()