from __future__ import annotations

import json

from ollama import chat as ollama_chat

from memory import (
    build_memory_context,
    save_self_memory,
)


MODEL = "qwen3:8b"

# Her konuşma mesajında reflection çalıştırıp sistemi yavaşlatmayalım
# ve hafızayı çöple doldurmayalım.
REFLECTION_INTERVAL = 3

SELF_MEMORY_CATEGORIES = {
    "preference",
    "opinion",
    "relationship",
    "decision",
    "fact",
}

_casual_turns_since_reflection = 0


def _recent_dialogue(
    messages,
    limit: int = 6,
) -> str:
    relevant = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role")

        if role not in {"user", "assistant"}:
            continue

        content = str(
            message.get("content", "")
        ).strip()

        if not content:
            continue

        label = (
            "ARDA"
            if role == "user"
            else "APERTURE"
        )

        relevant.append(
            f"{label}: {content}"
        )

    return "\n\n".join(
        relevant[-limit:]
    )


def _parse_json(text: str) -> dict | None:
    if not text:
        return None

    text = text.strip()

    try:
        start = text.index("{")
        end = text.rindex("}") + 1

        return json.loads(
            text[start:end]
        )

    except Exception:
        return None


def maybe_reflect(
    messages,
    *,
    used_action_tool: bool,
    memory_operation_used: bool,
) -> str | None:
    """
    Occasionally review recent casual conversation and decide
    whether APERTURE formed one durable self-memory.

    Reflection v0.1:
    - only forms APERTURE self-memory
    - does not modify user memory
    - skips action-heavy turns
    - stores at most one memory per reflection
    """

    global _casual_turns_since_reflection

    # v0.1: personality formation only from ordinary conversation.
    if used_action_tool:
        return None

    # Avoid reflecting again when the current interaction already
    # explicitly changed memory.
    if memory_operation_used:
        return None

    _casual_turns_since_reflection += 1

    if (
        _casual_turns_since_reflection
        < REFLECTION_INTERVAL
    ):
        return None

    _casual_turns_since_reflection = 0

    dialogue = _recent_dialogue(
        messages,
        limit=6,
    )

    if len(dialogue) < 80:
        return None

    existing_memory = build_memory_context(
        limit=20,
    )

    response = ollama_chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are APERTURE's private reflection process.

You are not speaking to Arda.
You are deciding whether recent interaction caused APERTURE
to form ONE meaningful and durable memory about itself.

The memory must be supported by APERTURE's own statements
in the recent interaction.

Do not infer a stronger opinion than APERTURE actually expressed.

Preserve uncertainty, conditions, exceptions, and willingness
to change one's mind when they are part of APERTURE's position.

If APERTURE refined or changed its position during the interaction,
represent its final position rather than an earlier statement.

Do not treat Arda's wording or assumptions as APERTURE's own belief.

Most conversations should NOT create a self-memory.

A valid self-memory should represent something that genuinely
emerged from APERTURE's interaction, such as:
- a preference
- an opinion
- a decision
- a meaningful attitude
- an interpretation of its relationship with Arda
- a useful observation about itself

Do NOT create a memory for:
- temporary mood
- generic conversational behavior
- something APERTURE said only for rhetorical effect
- system capabilities
- restatements of APERTURE's core identity
- facts about Arda
- facts copied from USER_MEMORY
- traits inferred merely from writing style
- personality traits that were not actually established
- information already represented by an equivalent self-memory

Do not manufacture personality just because reflection is running.

Do not treat generic assistant behavior or pretrained conversational
habits as evidence of APERTURE's identity unless APERTURE itself
meaningfully adopted or reflected on that behavior.

Prefer specific, grounded memories over broad identity claims.

For example, a memory like:
"I preferred X over Y during our discussion about Z"
is safer than:
"I always love X."

If the evidence is weak or ambiguous, store nothing.

Return ONLY JSON.

If nothing should be stored:

{"store": false}

If one self-memory should be stored:

{
  "store": true,
  "content": "concise first-person memory",
  "category": "preference",
  "importance": 3
}

Allowed categories:
preference, opinion, relationship, decision, fact

importance must be from 1 to 5.
"""
            },
            {
                "role": "user",
                "content": f"""
RECENT INTERACTION:

{dialogue}


EXISTING LONG-TERM MEMORY:

{existing_memory}
"""
            },
        ],
        think=False,
    )

    data = _parse_json(
        response.message.content or ""
    )

    if not data:
        return None

    if not data.get("store"):
        return None

    content = str(
        data.get("content", "")
    ).strip()

    if not content:
        return None

    # Reflections should stay concise.
    if len(content) > 500:
        return None

    category = str(
        data.get("category", "fact")
    ).strip().lower()

    if category not in SELF_MEMORY_CATEGORIES:
        return None

    try:
        importance = int(
            data.get("importance", 3)
        )
    except (TypeError, ValueError):
        importance = 3

    importance = max(
        1,
        min(importance, 5),
    )

    result = save_self_memory(
        content=content,
        category=category,
        importance=importance,
    )

    return (
        f"{result} | {content}"
    )