# APERTURE Cognitive Boundaries

This document defines architectural invariants for APERTURE's
cognitive systems.

These rules exist to prevent debugging fixes from gradually
turning APERTURE into a hand-scripted personality or
phrase-driven state machine.


## 1. Model and agent are different things

No single LLM inference is APERTURE.

APERTURE is the persistent system formed by:

- actor
- persona
- memory
- experience
- reflection
- dynamic self
- relationship state
- tools
- future cognitive processes

Internal cognitive modules are processes within APERTURE,
not separate characters or miniature APERTURE instances.


## 2. Code does not decide semantic meaning

Code may enforce:

- evidence ownership
- authority boundaries
- schemas
- allowed fields
- types
- length limits
- provenance
- lifecycle
- persistence rules
- permissions
- temporal validity
- destructive-action safety

Code should NOT decide that a sentence means:

- preference
- interest
- belief
- relationship significance
- closeness
- curiosity
- identity
- disagreement

through growing phrase lists or lexical trigger libraries.

Semantic interpretation belongs to cognitive model passes.


## 3. Lexical routing must never define cognition

Performance optimizations may eventually avoid unnecessary
model calls.

However, an optimization gate must not determine whether
a cognitive state is semantically possible.

Bad:

    "i prefer" -> preference analysis
    "our conversations" -> relationship analysis

Good:

    evidence
        -> semantic cognitive process
        -> structured result or null
        -> code validation


## 4. Evidence, authority, and output are separate

Every internal process should define:

### Identity scope

What process is this?

Example:

    An internal analysis process within APERTURE.

### Evidence scope

What information may establish its conclusions?

### Authority scope

What may the process decide or change?

### Output scope

What exact structured representation may it return?


## 5. Derived cognition is not independent evidence

Memory, Dynamic Self, Relationship State, and other derived
representations may influence APERTURE's future responses.

If APERTURE later repeats information because it appeared in
prior cognitive context, that repetition must not automatically
be treated as independent confirmation of the original cognition.

In particular:

    derived state
        -> actor response
        -> same derived state

must not become a self-reinforcing evidence loop.

Existing cognition may provide continuity.

It must not manufacture additional epistemic certainty merely
by causing APERTURE to repeat itself.


## 6. Memory is not current state

Long-term memory records durable continuity.

Dynamic Self represents what appears currently active.

A historical self-memory does not automatically become current
Dynamic Self.

It may become current again when present evidence supports it.


## 7. Relationship state has one owner

Current Arda-APERTURE interaction state belongs to the
Relationship Model.

Reflection must not independently create a parallel
relationship interpretation.

Future durable relationship memory should be produced through
an explicit consolidation path from relationship experience,
not through competing cognitive modules.


## 8. Temporary cognition is not identity

Dynamic Self and Relationship State are temporary,
fallible representations.

They are:

- context
- evidence
- revisable

They are not:

- commands
- permanent personality definitions
- behavioral scripts


## 9. Tests validate behavior, not trigger phrases

Regression tests should ask whether the architecture produces
the correct semantic outcome.

Tests should not require implementation details such as:

    phrase X must trigger module Y

because doing so encourages phrase-list growth.

Prefer invariants such as:

    ordinary user fact -> no relationship state

    explicit current self-position -> valid Dynamic Self candidate

    assistant compliance -> no durable self-memory

    old self-memory alone -> cannot create current Dynamic Self

    relationship state -> exact schema, no hidden scalar fields


## 10. Prefer architecture over prompt accumulation

When a semantic failure repeats:

1. locate the failing stage using provenance/debug traces
2. check evidence ownership
3. check process authority
4. check context contamination
5. check lifecycle
6. only then adjust semantic instructions

Do not automatically solve new failures by adding another
phrase, blacklist item, or special-case detector.


## Core Rule

Code constrains what cognition is allowed to use and change.

Models interpret what the evidence means.