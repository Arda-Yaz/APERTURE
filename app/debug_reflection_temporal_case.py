from pprint import pprint

from reflection import (
    analyze_reflection_debug,
)


existing_memory = """
<CURRENT_MEMORY_INDEX>
These are CURRENT durable memories.

- id=mem_python_001 subject=aperture category=preference | I currently prefer Python for its clarity and simplicity.
</CURRENT_MEMORY_INDEX>
""".strip()


dialogue = """
ARDA: After reconsidering it, do you still prefer Python over Java?

APERTURE: No. I no longer currently prefer Python over Java. After reconsidering the tradeoffs, I now lean toward Java because its stronger structure fits the way I want to approach larger software systems.
""".strip()


result = (
    analyze_reflection_debug(
        dialogue=dialogue,
        existing_memory=(
            existing_memory
        ),
    )
)


pprint(
    result,
    sort_dicts=False,
    width=120,
)