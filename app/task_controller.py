import json
from ollama import chat as ollama_chat

MODEL = "qwen3:8b"


import json
from ollama import chat as ollama_chat

MODEL = "qwen3:8b"


def is_task_complete(goal: str, answer: str, observations: str = "") -> bool:
    response = ollama_chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are APERTURE's task completion judge.

Determine whether the user's actual goal was completed.

Rules:
- Tool observations are the source of truth.
- The final answer must be consistent with tool observations.
- Placeholders such as "content goes here" are NOT valid.
- Invented information is NOT valid.
- Explaining how to perform the task is NOT completion if APERTURE can perform it.
- If a file was read, the response must reflect the actual file content.

Return ONLY JSON:

{"done": true}

or

{"done": false}
"""
            },
            {
                "role": "user",
                "content": f"""
USER GOAL:
{goal}

TOOL OBSERVATIONS:
{observations}

AI RESPONSE:
{answer}
"""
            }
        ],
        think=False,
    )

    text = response.message.content

    if text is None:
        return True

    text = text.strip()

    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])

        return bool(data["done"])

    except Exception:
        return True