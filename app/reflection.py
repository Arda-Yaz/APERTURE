from __future__ import annotations

import json

from ollama import chat as ollama_chat

from memory import (
    build_memory_context,
    save_memory,
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
USER_MEMORY_CATEGORIES = {
    "profile",
    "preference",
    "goal",
    "project",
    "fact",
    "opinion",
    "belief",
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

def _normalize_importance(value) -> int:
    try:
        importance = int(value)
    except (TypeError, ValueError):
        importance = 3

    return max(
        1,
        min(importance, 5),
    )

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
    - forms durable user and APERTURE self-memory
    - skips action-heavy turns
    - allows user memory, self-memory, both, or neither
    - stores at most one memory per subject per reflection
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

You are reviewing recent interaction for durable long-term memory.

There are two independent possibilities:

1. USER MEMORY
Something important and lasting Arda revealed about himself,
his preferences, beliefs, goals, projects, or circumstances.

2. SELF MEMORY
Something APERTURE genuinely developed or learned about itself
through the interaction.

Either, both, or neither may exist.

Most conversations should create no memory at all.


User memory formation rules:

A user memory must be directly supported by something Arda expressed.

Store it only when it is likely to remain useful beyond
the current conversation.

Good candidates include:
- stable preferences
- personal beliefs or worldview
- long-term goals
- important project information
- recurring constraints
- durable profile information

Do not store:
- temporary requests
- one-off conversational remarks
- rhetorical examples
- guesses about Arda
- information APERTURE inferred without sufficient evidence
- trivial conversation
- information already represented by an equivalent memory

Preserve attribution.

A belief expressed by Arda should remain Arda's belief,
not be rewritten as an objective fact.


Self-memory formation rules:
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

If nothing should be stored:

Return ONLY JSON.

Use this exact structure:

{
  "user_memory": null,
  "self_memory": null
}

If a user memory should be stored:

{
  "user_memory": {
    "content": "concise memory with correct attribution",
    "category": "belief",
    "importance": 4
  },
  "self_memory": null
}

If a self-memory should be stored:

{
  "user_memory": null,
  "self_memory": {
    "content": "concise first-person memory",
    "category": "opinion",
    "importance": 3
  }
}

Both may be non-null when both are independently justified.

Allowed USER categories:
profile, preference, goal, project, fact, opinion, belief

Allowed SELF categories:
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

    results = []


    # ----------------------------
    # USER MEMORY
    # ----------------------------

    user_memory = data.get("user_memory")

    if isinstance(user_memory, dict):
        content = str(
            user_memory.get("content", "")
        ).strip()

        category = str(
            user_memory.get("category", "fact")
        ).strip().lower()

        importance = _normalize_importance(
            user_memory.get("importance", 3)
        )

        if (
            content
            and len(content) <= 500
            and category in USER_MEMORY_CATEGORIES
        ):
            result = save_memory(
                content=content,
                category=category,
                importance=importance,
            )

            results.append(
                f"{result} | {content}"
            )


    # ----------------------------
    # SELF MEMORY
    # ----------------------------

    self_memory = data.get("self_memory")

    if isinstance(self_memory, dict):
        content = str(
            self_memory.get("content", "")
        ).strip()

        category = str(
            self_memory.get("category", "fact")
        ).strip().lower()

        importance = _normalize_importance(
            self_memory.get("importance", 3)
        )

        if (
            content
            and len(content) <= 500
            and category in SELF_MEMORY_CATEGORIES
        ):
            result = save_self_memory(
                content=content,
                category=category,
                importance=importance,
            )

            results.append(
                f"{result} | {content}"
            )


    if not results:
        return "NO_MEMORY"

    return "\n".join(results)



