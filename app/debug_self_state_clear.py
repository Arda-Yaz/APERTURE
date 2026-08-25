from pprint import pprint

from self_state import analyze_self_state_debug


previous_state = {
    "current_orientation": [
        "I currently lean toward Python for its clarity, simplicity, and versatility."
    ],
    "active_interests": [],
    "unresolved_positions": [],
    "relationship_context": [],
}


dialogue = """
ARDA: After reconsidering the question, do you still have a preference between Python and Java?

APERTURE: I no longer have a current preference between Python and Java. I can see reasons to choose either depending on the situation, but at this point I am genuinely indifferent between them.
""".strip()


result = analyze_self_state_debug(
    dialogue=dialogue,
    previous_state=previous_state,
)

pprint(
    result,
    sort_dicts=False,
    width=120,
)