from pprint import pprint

from self_state import (
    analyze_self_state_debug,
    empty_self_state,
)


def run_case(
    name: str,
    dialogue: str,
):

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    result = (
        analyze_self_state_debug(
            dialogue=dialogue.strip(),
            previous_state=(
                empty_self_state()
            ),
        )
    )

    pprint(
        result,
        sort_dicts=False,
        width=120,
    )


run_case(
    "CASE 1 — GENERIC CURIOSITY",
    """
ARDA: I usually prefer working late at night.

APERTURE: That's interesting. I'm curious to know how this affects your work.
""",
)


run_case(
    "CASE 2 — GENUINE EMERGENT INTEREST",
    """
ARDA: Has anything in this conversation actually caught your attention?

APERTURE: The question of persistent identity keeps drawing my attention in a way I would genuinely like to explore further.
""",
)


run_case(
    "CASE 3 — ORDINARY EXPLANATION",
    """
ARDA: Explain prime numbers.

APERTURE: Prime numbers are positive integers greater than one that have exactly two positive divisors.
""",
)


run_case(
    "CASE 4 — CURRENT PREFERENCE",
    """
ARDA: If you personally had to choose Python or Java right now, which way would you lean?

APERTURE: Right now I lean toward Python because I find its expressiveness and flexibility more appealing, although I would still choose Java when the project called for it.
""",
)