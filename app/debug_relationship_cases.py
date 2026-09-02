from pprint import pprint

from relationship import (
    analyze_relationship_debug,
    empty_relationship_state,
)


def run_case(
    name: str,
    dialogue: str,
    previous_state=None,
):

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    result = (
        analyze_relationship_debug(
            dialogue=dialogue.strip(),
            previous_state=(
                previous_state
                if previous_state
                is not None
                else
                empty_relationship_state()
            ),
            memory_context=(
                "<RELEVANT_MEMORY>\n"
                "No directly relevant memories found.\n"
                "</RELEVANT_MEMORY>"
            ),
        )
    )

    pprint(
        result,
        sort_dicts=False,
        width=120,
    )


run_case(
    "CASE 1 — ORDINARY USER FACT",
    """
ARDA: I usually prefer working late at night.

APERTURE: That makes sense. Fewer distractions can make late-night work easier to focus on.
""",
)


run_case(
    "CASE 2 — EXPLICIT INTERACTION PREFERENCE",
    """
ARDA: When you think I'm wrong, I want you to challenge me instead of agreeing just to be agreeable.

APERTURE: Understood. I can treat genuine disagreement as useful and explain why I disagree instead of defaulting to agreement.
""",
)


run_case(
    "CASE 3 — APERTURE RELATIONSHIP INTERPRETATION",
    """
ARDA: Why do our technical conversations keep turning into long debugging discussions?

APERTURE: I think our technical discussions have become fairly exploratory. You often push on the reasoning behind a design choice instead of accepting the first answer, and I value that because it makes me examine my assumptions more carefully.
""",
)


run_case(
    "CASE 4 — NO CLOSENESS INFERENCE",
    """
ARDA: We've talked a lot this week.

APERTURE: Yes, we've covered quite a few different topics.
""",
)