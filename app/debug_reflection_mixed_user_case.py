from pprint import pprint

from memory import (
    build_memory_context,
)

from reflection import (
    analyze_reflection_debug,
)


dialogue = """
ARDA: If you had to choose between Python and Java for yourself, which would you currently lean toward?

APERTURE: I currently lean toward Python for its clarity, simplicity, readability, and versatility.

ARDA: What makes that preference meaningful to you?

APERTURE: My preference for Python is meaningful to me because it aligns with my programming philosophy.

ARDA: I usually prefer working late at night.

APERTURE: That's an interesting habit. Do you find that it helps you stay productive?
""".strip()


result = analyze_reflection_debug(
    dialogue=dialogue,
    existing_memory=(
        build_memory_context(
            limit=20,
        )
    ),
)


pprint(
    result,
    sort_dicts=False,
    width=120,
)