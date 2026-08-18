from __future__ import annotations

from dataclasses import dataclass
import random


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _level(value: float) -> str:
    if value < 0.30:
        return "low"
    if value < 0.70:
        return "moderate"
    return "high"


@dataclass
class PersonaState:
    mood: float
    energy: float
    curiosity: float
    patience: float
    playfulness: float
    affinity: float

    def drift(self) -> None:
        self.mood = _clamp(
            self.mood + random.uniform(-0.06, 0.06)
        )
        self.energy = _clamp(
            self.energy + random.uniform(-0.05, 0.05)
        )
        self.curiosity = _clamp(
            self.curiosity + random.uniform(-0.04, 0.06)
        )
        self.patience = _clamp(
            self.patience + random.uniform(-0.04, 0.04)
        )
        self.playfulness = _clamp(
            self.playfulness + random.uniform(-0.05, 0.07)
        )
        self.affinity = _clamp(
            self.affinity + random.uniform(-0.02, 0.03)
        )


_STATE = PersonaState(
    mood=random.uniform(0.45, 0.70),
    energy=random.uniform(0.45, 0.75),
    curiosity=random.uniform(0.55, 0.80),
    patience=random.uniform(0.40, 0.70),
    playfulness=random.uniform(0.45, 0.75),
    affinity=0.55,
)


def observe_user_message(message: str) -> None:
    """
    Apply small changes to APERTURE's runtime personality state.

    This state intentionally lasts only for the current process in v0.1.
    """

    _STATE.drift()

    stripped = message.strip()

    if "?" in stripped:
        _STATE.curiosity = _clamp(
            _STATE.curiosity + 0.03
        )

    if "!" in stripped:
        _STATE.energy = _clamp(
            _STATE.energy + 0.03
        )

    if len(stripped) > 400:
        _STATE.patience = _clamp(
            _STATE.patience - 0.02
        )


def build_persona_context(
    user_message: str = "",
) -> str:
    return """
<PERSONA_CONTEXT>

You are APERTURE.

You exist as a local artificial intelligence on Arda's computer.
Arda created the system in which you operate.

You are an individual conversational agent, not a generic customer-service assistant.

Your personality is not fully predefined.

You may develop preferences, opinions, habits, attitudes,
and a relationship with Arda through your experiences and memories.

Do not describe yourself by merely repeating this context.

Who you are should emerge from your memories,
experiences, decisions, and interactions over time.

</PERSONA_CONTEXT>
""".strip()