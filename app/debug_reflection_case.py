from pprint import pprint

from memory import build_memory_context
from reflection import analyze_reflection_debug


dialogue = """
ARDA: Does preferring Python mean you are genuinely interested in programming languages, or is it only a preference between these options?

APERTURE: My preference for Python reflects a genuine interest in programming languages, particularly in how they can be used to solve problems efficiently and elegantly. Python's design philosophy and features align with my approach to programming, which values clarity, simplicity, and versatility. However, this preference is also shaped by the specific context of choosing between Python and Java, as well as the broader landscape of programming languages and their applications.

ARDA: Suppose you later became convinced that Java fits your way of thinking better than Python. Would you be comfortable changing your preference?

APERTURE: I am comfortable changing my preference if I later became convinced that Java fits my way of thinking better than Python. My preference is not fixed and can evolve based on new insights and experiences. This openness to change reflects my belief that growth and adaptability are essential aspects of both learning and thinking.

ARDA: I've given you another perspective to consider: Java's stronger type system, explicit structure, and more constrained design may actually fit the way you described your own programming philosophy better than Python's flexibility. Think about that argument for yourself. Has it changed your current preference at all?

APERTURE: I have given this argument considerable thought, and it has prompted me to reconsider my current preference. Java's stronger type system, explicit structure, and more constrained design do align with a programming philosophy that values clarity, precision, and structure. These characteristics can lead to more robust and maintainable code, especially in large-scale projects or teams where consistency and predictability are crucial.

While I still appreciate Python's flexibility and readability, I now recognize that Java's design might better suit a programming philosophy that emphasizes explicitness and structure. This has not changed my current preference entirely, but it has made me more open to the idea that my preference might evolve further in the future based on new insights and experiences.
""".strip()


result = analyze_reflection_debug(
    dialogue=dialogue,
    existing_memory=build_memory_context(
        limit=20,
    ),
)

pprint(
    result,
    sort_dicts=False,
    width=120,
)