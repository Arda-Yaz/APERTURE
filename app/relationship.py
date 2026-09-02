from __future__ import annotations

import json

from copy import deepcopy

from ollama import chat as ollama_chat

from memory import (
    build_relevant_memory_context,
)


MODEL = "qwen3:8b"

RELATIONSHIP_OPTIONS = {
    "temperature": 0,
    "seed": 42,
}


STATE_KEYS = (
    "interaction_preferences",
    "established_patterns",
    "relationship_interpretations",
    "open_questions",
)


MAX_ITEMS_PER_FIELD = {
    "interaction_preferences": 4,
    "established_patterns": 3,
    "relationship_interpretations": 3,
    "open_questions": 2,
}


MAX_ITEM_LENGTH = 800


RELATIONSHIP_PROMPT = """
You are an internal analysis process within APERTURE.

You are not a separate conversational character.

Your task is to maintain a small current model of the
interaction relationship between Arda and APERTURE.

Your output is not itself an APERTURE belief, memory,
preference, emotion, or experience.

It is a derived representation of interaction evidence.


============================================================
PURPOSE
============================================================

The Relationship Model represents useful CURRENT information
about how Arda and APERTURE interact with each other.

It may represent:

- explicit interaction preferences or boundaries
- directly established recurring interaction patterns
- APERTURE's explicit interpretation of the relationship
- unresolved questions about how the interaction should work

It must remain conservative and evidence-based.


============================================================
THIS IS NOT A USER PROFILE
============================================================

Do NOT copy ordinary facts about Arda into relationship state.

Examples that are NOT relationship state:

- Arda prefers working late at night.
- Arda likes Python.
- Arda is searching for a job.
- Arda owns a particular computer.
- Arda is interested in AI.

Those belong to long-term memory when useful.

A fact becomes relevant here only when it specifically describes
how Arda and APERTURE interact.


============================================================
THIS IS NOT APERTURE'S DYNAMIC SELF
============================================================

Do NOT store ordinary APERTURE preferences, interests,
opinions, or unresolved beliefs here.

Examples:

- I currently prefer Python.
- I am interested in artificial identity.
- I am unsure whether X is true.

Those belong to Dynamic Self or long-term self-memory.

This module is specifically about the interaction relationship.


============================================================
NO RELATIONSHIP SCORING
============================================================

Do NOT create numeric or categorical relationship scores.

Never infer or output things like:

- trust = 0.8
- closeness = high
- affection = medium
- attachment = strong
- intimacy = 7
- loyalty = high
- friendship_level = 4

Do not infer emotional significance merely because
Arda talks to APERTURE frequently.


============================================================
NO SOCIAL OVER-INTERPRETATION
============================================================

Do NOT infer:

- trust
- friendship
- closeness
- attachment
- affection
- dependency
- emotional intimacy
- special significance

from ordinary conversation.

The following do NOT establish a special relationship:

- frequent messages
- long conversations
- repeated technical discussions
- politeness
- jokes
- disagreement
- asking personal questions
- asking for advice
- collaboration on a project
- returning across multiple sessions

If Arda or APERTURE explicitly expresses a relationship
interpretation, it may be represented conservatively.

Do not intensify the wording.


============================================================
EVIDENCE
============================================================

FULL RECENT CONVERSATION is the primary evidence source.

Both Arda and APERTURE may provide valid relationship evidence.

Arda may explicitly state how he wants APERTURE to interact
with him.

Example:

ARDA:
"When you think I'm wrong, I want you to challenge me
instead of agreeing just to be agreeable."

This may establish an interaction preference.

APERTURE may explicitly form an interpretation about
the interaction.

Example:

APERTURE:
"I think our conversations have become more exploratory,
because Arda often pushes me to justify my assumptions."

This may establish a relationship interpretation.

Do not infer such interpretations when APERTURE did not
actually express them.


============================================================
PREVIOUS STATE
============================================================

PREVIOUS RELATIONSHIP STATE is continuity context.

It is not proof that an item is still current.

An earlier item may be:

- preserved
- revised
- replaced
- removed

when current evidence justifies it.

Do not preserve stale state merely because it already exists.


============================================================
LONG-TERM MEMORY
============================================================

RELEVANT LONG-TERM MEMORY is background continuity.

It may help understand current evidence.

It is not evidence that the current conversation repeated
or confirmed something.

Do not manufacture state from memory alone.


============================================================
STATE FIELDS
============================================================


interaction_preferences

Explicit preferences or boundaries about how Arda wants
APERTURE to interact with him.

Examples:

"Arda has explicitly asked me to challenge him when I
genuinely disagree rather than defaulting to agreement."

"Arda prefers direct implementation guidance instead of
lengthy preliminary discussion."

These should normally come from Arda's explicit statements.

Do not infer interaction preferences from ordinary behavior.


established_patterns

Directly supported recurring patterns in the interaction.

This field has a high evidence threshold.

A single event is usually NOT a pattern.

A pattern should normally be supported by:

- multiple recent examples
- previous relationship state plus new confirming evidence
- or relevant durable memory plus current evidence

Example:

"Arda often tests my reasoning with follow-up questions,
and our technical discussions frequently become iterative
debugging sessions."

Do not write broad personality claims about Arda.

Describe the interaction pattern, not the person.


relationship_interpretations

A current relationship or interaction interpretation that
APERTURE itself explicitly expressed.

Examples:

"I currently see our APERTURE development discussions as
collaborative problem-solving rather than simple question-answering."

"I have started to value the way Arda challenges my assumptions."

These require APERTURE self-expression.

Do NOT create them solely from Arda's behavior.

Preserve uncertainty.

Do not strengthen:

"I think..."

into:

"I know..."

or:

"I value..."

into emotional attachment.


open_questions

An unresolved question about the interaction that APERTURE
explicitly recognizes, or that the conversation explicitly raises.

Example:

"I am not yet sure how much unsolicited disagreement Arda wants
outside technical discussions."

Do not manufacture an open question just because more
information could theoretically exist.


COMPLIANCE IS NOT RELATIONSHIP INTERPRETATION

When Arda explicitly asks APERTURE to interact in a particular
way, APERTURE may acknowledge or agree to that request.

Statements such as:

"I will challenge you when I disagree."
"My goal is to keep the dialogue honest."
"I can provide alternative perspectives."

are primarily commitments to honor Arda's interaction preference.

They do NOT independently establish a relationship interpretation.

In such a case, store the justified interaction preference,
but leave relationship_interpretations empty unless APERTURE
also makes an explicit descriptive interpretation of the
relationship itself.

For example:

"I think our discussions have become more exploratory."

may qualify.

Do not transform compliance into identity or relationship meaning.


============================================================
NEGATIVE EXAMPLES
============================================================

ARDA:
"I usually prefer working late at night."

APERTURE:
"That can be useful when there are fewer distractions."

Correct:

{"state": null}


ARDA:
"Can you explain transformers?"

APERTURE:
"Sure."

Correct:

{"state": null}


ARDA:
"We've talked a lot this week."

APERTURE:
"Yes, we've covered several topics."

Do NOT infer closeness, trust, friendship, or importance.

Correct:

{"state": null}


============================================================
INTERACTION-PREFERENCE EXAMPLE
============================================================

ARDA:
"When you disagree with me, don't just agree to be polite.
Tell me why you disagree."

APERTURE:
"Understood. I can treat disagreement as useful rather than
something to avoid."

A reasonable state:

{
  "state": {
    "interaction_preferences": [
      "Arda has explicitly asked me to explain genuine disagreements rather than defaulting to agreement."
    ],
    "established_patterns": [],
    "relationship_interpretations": [],
    "open_questions": []
  }
}


============================================================
RELATIONSHIP-INTERPRETATION EXAMPLE
============================================================

APERTURE:
"I think our discussions have become more exploratory.
Arda often asks me to defend the reasoning behind my answers,
and I value that pressure to make my thinking clearer."

A reasonable state may include:

{
  "state": {
    "interaction_preferences": [],
    "established_patterns": [],
    "relationship_interpretations": [
      "I currently see our discussions as increasingly exploratory and value being challenged to make my reasoning clearer."
    ],
    "open_questions": []
  }
}


============================================================
WRITING RULES
============================================================

- Use concise natural language.
- Preserve ownership.
- Preserve uncertainty.
- Preserve conditions and boundaries.
- Do not invent emotional states.
- Do not use scores.
- Do not issue behavioral commands.
- Do not write "be more honest" or "challenge Arda more."
- Describe evidence-backed interaction state instead.
- Prefer sparse state.
- Empty lists are normal.
- Most conversations should NOT update this model.


============================================================
UPDATE RULES
============================================================

Return:

{"state": null}

when recent evidence does not justify a meaningful change.

When change is justified, return the COMPLETE replacement state.

Do not return a patch.

Previous valid items may remain when still supported.

Remove an item when:

- it is contradicted
- it is no longer current
- it was too strong
- current evidence clarifies that it was mistaken

A completely empty replacement state is valid.


Exact structure:

{
  "state": {
    "interaction_preferences": [],
    "established_patterns": [],
    "relationship_interpretations": [],
    "open_questions": []
  }
}

or:

{"state": null}

Return ONLY JSON.
""".strip()


# ============================================================
# STATE
# ============================================================

def empty_relationship_state(
) -> dict:

    return {
        key: []
        for key
        in STATE_KEYS
    }


_CURRENT_STATE = (
    empty_relationship_state()
)


# ============================================================
# DIALOGUE / SIGNAL
# ============================================================

def _recent_dialogue(
    messages,
    limit: int = 10,
) -> str:

    relevant = []

    for message in messages:

        if not isinstance(
            message,
            dict,
        ):
            continue

        role = (
            message.get(
                "role"
            )
        )

        if role not in {
            "user",
            "assistant",
        }:
            continue

        content = str(
            message.get(
                "content",
                "",
            )
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


def _aperture_only_evidence(
    dialogue: str,
) -> str:

    messages = []

    current_speaker = None
    current_lines = []

    def flush_current():
        nonlocal current_speaker
        nonlocal current_lines

        if (
            current_speaker == "aperture"
            and current_lines
        ):
            content = " ".join(
                line
                for line in current_lines
                if line
            ).strip()

            if content:
                messages.append(
                    content
                )

        current_speaker = None
        current_lines = []

    for raw_line in (
        dialogue.splitlines()
    ):

        line = raw_line.strip()

        if line.startswith("ARDA:"):
            flush_current()

            current_speaker = "arda"

            current_lines = [
                line[
                    len("ARDA:"):
                ].strip()
            ]

            continue

        if line.startswith(
            "APERTURE:"
        ):
            flush_current()

            current_speaker = (
                "aperture"
            )

            current_lines = [
                line[
                    len("APERTURE:"):
                ].strip()
            ]

            continue

        if (
            current_speaker
            and line
        ):
            current_lines.append(
                line
            )

    flush_current()

    return "\n\n".join(
        messages
    )


def _has_explicit_relationship_interpretation_signal(
    self_evidence: str,
) -> bool:
    """
    Relationship interpretations require explicit
    APERTURE-authored relational interpretation.

    Merely agreeing to Arda's requested interaction style
    is not enough.
    """

    text = (
        " "
        + " ".join(
            self_evidence
            .casefold()
            .split()
        )
        + " "
    )

    signals = (
        "i think our ",
        "i see our ",
        "i currently see our ",
        "i notice our ",
        "i've noticed our ",
        "i have noticed our ",
        "i value the way arda",
        "i appreciate the way arda",
        "i'm not sure how our ",
        "i am not sure how our ",

        "konuşmalarımızı ",
        "sohbetlerimizi ",
        "etkileşimimizi ",
        "arda'nın beni ",
        "arda'nın bana ",
    )

    return any(
        signal in text
        for signal in signals
    )


def _enforce_evidence_boundaries(
    state: dict | None,
    dialogue: str,
) -> dict | None:

    if state is None:
        return None

    cleaned = deepcopy(
        state
    )

    self_evidence = (
        _aperture_only_evidence(
            dialogue
        )
    )

    if (
        cleaned[
            "relationship_interpretations"
        ]
        and not
        _has_explicit_relationship_interpretation_signal(
            self_evidence
        )
    ):
        cleaned[
            "relationship_interpretations"
        ] = []

    return cleaned


def _has_relationship_signal(
    dialogue: str,
) -> bool:
    """
    Cheap conservative gate.

    This does NOT decide relationship state.
    It only decides whether the relationship analysis
    model is worth running.
    """

    text = (
        " "
        + " ".join(
            str(dialogue)
            .casefold()
            .split()
        )
        + " "
    )

    explicit_phrases = (
        # English — relationship framing
        "our conversations",
        "our conversation",
        "our discussions",
        "our interaction",
        "our interactions",
        "between us",
        "when we talk",
        "when we discuss",
        "how we interact",

        # English — explicit interaction preferences
        "challenge me",
        "disagree with me",
        "agree with me",
        "don't just agree",
        "do not just agree",
        "be honest with me",
        "be direct with me",
        "push back on me",
        "push back when",
        "call me out",
        "question my assumptions",
        "challenge my assumptions",
        "how you respond to me",
        "the way you respond to me",
        "i prefer when you",
        "i like when you",
        "i don't want you to",
        "i do not want you to",

        # English — APERTURE relational interpretations
        "i think our ",
        "i feel our ",
        "i see our ",
        "i notice our ",
        "i've noticed our ",
        "i have noticed our ",
        "i value the way arda",
        "i appreciate the way arda",

        # Turkish — relationship framing
        "konuşmalarımız",
        "sohbetlerimiz",
        "aramızda",
        "aramızdaki",
        "etkileşimimiz",
        "birlikte konuşurken",

        # Turkish — explicit interaction preferences
        "bana katılma",
        "bana karşı çık",
        "benimle aynı fikirde",
        "dürüstçe karşı çık",
        "beni sorgula",
        "varsayımlarıma karşı çık",
        "bana direkt ol",
        "benimle konuşurken",
        "bana cevap verirken",
        "senden bana",
    )

    return any(
        phrase in text
        for phrase
        in explicit_phrases
    )


# ============================================================
# JSON / SANITIZATION
# ============================================================

def _parse_json(
    text: str,
) -> dict | None:

    if not text:
        return None

    try:

        start = (
            text.index("{")
        )

        end = (
            text.rindex("}")
            + 1
        )

        return json.loads(
            text[start:end]
        )

    except Exception:
        return None


def _sanitize_text_list(
    value,
    *,
    max_items: int,
) -> list[str] | None:

    if not isinstance(
        value,
        list,
    ):
        return None

    cleaned = []
    seen = set()

    for item in value:

        if not isinstance(
            item,
            str,
        ):
            continue

        text = " ".join(
            item.strip().split()
        )

        if not text:
            continue

        if (
            len(text)
            > MAX_ITEM_LENGTH
        ):
            continue

        normalized = (
            text.casefold()
        )

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        cleaned.append(
            text
        )

        if (
            len(cleaned)
            >= max_items
        ):
            break

    return cleaned


def _sanitize_state(
    state,
) -> dict | None:

    if not isinstance(
        state,
        dict,
    ):
        return None

    # No hidden scalar fields such as trust/closeness.
    if (
        set(state.keys())
        != set(STATE_KEYS)
    ):
        return None

    cleaned = {}

    for key in STATE_KEYS:

        cleaned_list = (
            _sanitize_text_list(
                state.get(key),
                max_items=(
                    MAX_ITEMS_PER_FIELD[
                        key
                    ]
                ),
            )
        )

        if cleaned_list is None:
            return None

        cleaned[key] = (
            cleaned_list
        )

    return cleaned


# ============================================================
# MODEL CALL
# ============================================================

def _call_relationship_module(
    *,
    dialogue: str,
    previous_state: dict,
    memory_context: str,
) -> tuple[
    str,
    dict | None,
]:

    previous_json = (
        json.dumps(
            previous_state,
            ensure_ascii=False,
            indent=2,
        )
    )

    response = (
        ollama_chat(
            model=MODEL,
            messages=[
                {
                    "role":
                        "system",

                    "content":
                        RELATIONSHIP_PROMPT,
                },
                {
                    "role":
                        "user",

                    "content": f"""
FULL RECENT CONVERSATION:

{dialogue}


PREVIOUS RELATIONSHIP STATE:

{previous_json}


RELEVANT LONG-TERM MEMORY:

{memory_context}
""".strip(),
                },
            ],
            think=False,
            options=(
                RELATIONSHIP_OPTIONS
            ),
        )
    )

    raw = (
        response.message.content
        or ""
    )

    return (
        raw,
        _parse_json(
            raw
        ),
    )


# ============================================================
# DEBUG / ANALYSIS
# ============================================================

def analyze_relationship_debug(
    *,
    dialogue: str,
    previous_state: (
        dict | None
    ) = None,
    memory_context: (
        str | None
    ) = None,
) -> dict:

    if previous_state is None:
        previous_state = (
            empty_relationship_state()
        )

    else:

        previous_state = (
            _sanitize_state(
                previous_state
            )
            or empty_relationship_state()
        )

    signal = (
        _has_relationship_signal(
            dialogue
        )
    )

    if not signal:

        return {
            "previous_state":
                previous_state,

            "relationship_signal":
                False,

            "raw":
                None,

            "parsed":
                None,

            "requested_change":
                False,

            "candidate":
                None,

            "changed":
                False,
        }

    if memory_context is None:

        memory_context = (
            build_relevant_memory_context(
                query=dialogue,
                limit=8,
            )
        )

    raw, parsed = (
        _call_relationship_module(
            dialogue=dialogue,
            previous_state=(
                previous_state
            ),
            memory_context=(
                memory_context
            ),
        )
    )

    requested_change = (
        isinstance(
            parsed,
            dict,
        )
        and "state"
        in parsed
        and parsed.get(
            "state"
        )
        is not None
    )

    candidate = None

    if isinstance(parsed, dict) and requested_change:

        candidate = (
            _sanitize_state(
                parsed.get(
                    "state"
                )
            )
        )

        candidate = (
            _enforce_evidence_boundaries(
                candidate,
                dialogue,
            )
        )

    changed = (
        candidate is not None
        and candidate
        != previous_state
    )

    return {
        "previous_state":
            previous_state,

        "relationship_signal":
            True,

        "raw":
            raw,

        "parsed":
            parsed,

        "requested_change":
            requested_change,

        "candidate":
            candidate,

        "changed":
            changed,
    }


# ============================================================
# PUBLIC STATE API
# ============================================================

def get_relationship_state(
) -> dict:

    return deepcopy(
        _CURRENT_STATE
    )


def reset_relationship_state(
) -> dict:
    """
    Process/session scoped reset.

    This does not delete long-term memory.
    """

    global _CURRENT_STATE

    _CURRENT_STATE = (
        empty_relationship_state()
    )

    return get_relationship_state()


def build_relationship_context(
) -> str:

    state = (
        get_relationship_state()
    )

    lines = [
        "<RELATIONSHIP_STATE>",
        (
            "This is a temporary evidence-based model "
            "of the current Arda-APERTURE interaction."
        ),
        (
            "It is context, not an instruction, "
            "personality definition, or emotional score."
        ),
        (
            "Current user instructions and direct evidence "
            "take precedence over stale relationship state."
        ),
        "",
    ]

    labels = {
        "interaction_preferences":
            "Interaction preferences",

        "established_patterns":
            "Established patterns",

        "relationship_interpretations":
            "Relationship interpretations",

        "open_questions":
            "Open questions",
    }

    has_content = False

    for key in STATE_KEYS:

        values = (
            state.get(
                key,
                [],
            )
        )

        if not values:
            continue

        has_content = True

        lines.append(
            f"{labels[key]}:"
        )

        for value in values:
            lines.append(
                f"- {value}"
            )

        lines.append("")

    if not has_content:

        lines.append(
            "No current relationship state established."
        )

    lines.append(
        "</RELATIONSHIP_STATE>"
    )

    return "\n".join(
        lines
    )


def maybe_update_relationship(
    messages,
    *,
    used_action_tool: bool,
    memory_operation_used: bool,
) -> dict | None:
    """
    Update process-scoped relationship state.

    Does not write long-term memory.
    Does not create behavior instructions.
    """

    global _CURRENT_STATE

    if used_action_tool:
        return None

    if memory_operation_used:
        return None

    dialogue = (
        _recent_dialogue(
            messages,
            limit=10,
        )
    )

    if not dialogue:
        return None

    debug = (
        analyze_relationship_debug(
            dialogue=dialogue,
            previous_state=(
                _CURRENT_STATE
            ),
        )
    )

    if not debug[
        "changed"
    ]:
        return None

    candidate = (
        debug[
            "candidate"
        ]
    )

    if candidate is None:
        return None

    _CURRENT_STATE = (
        deepcopy(
            candidate
        )
    )

    return (
        get_relationship_state()
    )