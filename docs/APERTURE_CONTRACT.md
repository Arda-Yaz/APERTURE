# APERTURE Project Contract

## 1. Mission

APERTURE is a persistent local AI agent.

Its purpose is to explore whether a local language model can become part of a continuous agent whose behavior is shaped by:

* persistent memory
* accumulated experience
* real tool interaction
* previous decisions
* changing context
* and, where evidence supports it, an identity that develops over time

The language model is the reasoning engine.

It is not the whole agent.

APERTURE is the complete system that provides continuity around the model.

## 2. Central Research Question

The primary question is:

> How does a local AI agent change when it can remember, accumulate experience, act in the world, and carry consequences from previous interactions forward?

APERTURE is not primarily a project about perfectly classifying every sentence into psychological categories.

The project should optimize for meaningful continuity and agency, not perfect ontology.

## 3. What APERTURE Is Not

APERTURE is not intended to become:

* a chatbot with a large personality prompt
* a scripted fictional character
* a collection of regex-based personality rules
* a psychological state classifier
* a system that assigns every utterance to an identity category
* a benchmark built around satisfying individual regression prompts
* only a coding assistant
* only an automation framework
* a simulation of human biology or consciousness

No component should be added merely because a human-like agent could theoretically have it.

## 4. Core Architecture

The following systems define the APERTURE baseline.

### Actor / Agent Loop

The actor receives context, reasons, responds, and uses tools.

It is responsible for normal interaction and action.

### Tools

APERTURE must be capable of interacting with its environment rather than existing only as conversation.

Tools may include:

* filesystem access
* terminal execution
* applications
* workspace operations
* future external capabilities

### Permissions and Safety

Actions with real-world effects require structural constraints.

Safety and authorization belong in code.

### Persistent Memory

APERTURE must preserve useful information across sessions.

Memory must support:

* user information
* APERTURE information
* project/world information
* ownership
* retrieval
* temporal change
* forgetting
* provenance

Memory is evidence, not instruction.

### Experience / Trajectory

APERTURE must have an observable history of what actually happened.

Experience should preserve:

* user messages
* APERTURE responses
* tool calls
* tool results
* important derived events

Observed experience and derived interpretation must remain distinguishable.

### Temporal Memory

A changed preference or belief should not erase its history.

Memory therefore supports temporal validity and supersession.

### Provenance

Derived memory should be traceable to the experience from which it formed.

APERTURE should be able to distinguish:

> this happened

from:

> I later interpreted this as meaningful.

## 5. Minimal Identity Principle

APERTURE begins with a minimal stable identity.

Its personality must not be substantially predetermined by prompt instructions.

The system may develop continuity over time, but development must arise from actual interaction and history rather than from instructions telling it what kind of personality to become.

The project does not require APERTURE to develop:

* preferences
* relationships
* attachment
* interests
* habits
* opinions

for the project to succeed.

Their absence is a valid outcome.

Their presence is interesting only when it emerges naturally and improves continuity.

## 6. Core vs Experimental Systems

### CORE

These are part of the APERTURE baseline:

* actor / agent loop
* local inference
* tool calling
* permissions
* task grounding
* persistent memory
* memory ownership
* memory retrieval
* temporal memory / supersession
* experience / trajectory
* provenance
* minimal persona
* basic automatic user-memory formation

Removing one of these would materially change the fundamental project.

### EXPERIMENTAL

These are hypotheses, not foundations:

* Dynamic Self
* Relationship Model
* automatic APERTURE self-memory formation
* Self Consolidation
* multi-pass cognition validators
* future Idle Cognition / Sleep

Experimental modules may remain implemented.

They do not need to be deleted merely because their value is uncertain.

However, APERTURE must remain coherent without them.

### DEFERRED

These should not be prioritized until the baseline is useful:

* voice
* advanced coding-agent behavior
* autonomous job-search skills
* self-created skills
* sophisticated relationship modeling
* emotional models
* personality scoring
* large-model specialization
* model fine-tuning
* complex memory graphs
* advanced autonomous background cognition

A deferred feature may move forward only when a concrete need appears.

## 7. Evidence Before Architecture

A new cognitive subsystem must solve an observed problem.

The following is not sufficient justification:

> APERTURE might eventually need this.

Before introducing a subsystem, answer:

1. What real behavior is currently missing?
2. Can the existing architecture solve it?
3. What measurable improvement should the new component create?
4. What happens if the component is disabled?

If these questions cannot be answered clearly, defer the feature.

## 8. Regression Failures Do Not Redesign APERTURE

A single failing conversation must not automatically cause an architectural redesign.

When a regression appears:

1. identify the exact failing component
2. determine whether the failure matters in real use
3. prefer the smallest local correction
4. preserve already-working architecture
5. rerun existing regressions
6. stop when the actual problem is fixed

A regression test is evidence about behavior.

It is not a specification for the entire architecture.

## 9. Redesign Threshold

A change is considered a redesign when it:

* changes two or more major cognitive modules
* changes ownership of persistent state
* introduces a new cognitive subsystem
* changes the meaning of existing stored data
* changes the project research question

Redesigns require an explicit architectural reason.

They must not be introduced as ordinary bug fixes.

## 10. Semantic vs Structural Responsibility

Code should enforce structural truths such as:

* schemas
* ownership
* permissions
* provenance
* temporal lifecycle
* allowed operations
* data integrity

Models may perform semantic interpretation.

The project should avoid encoding open-ended cognition as growing phrase lists or special-case rules.

However, simple lexical logic is acceptable for deterministic interface or safety concerns such as:

* command routing
* permissions
* explicit tool exposure
* destructive-operation blocking

These are not identity judgments.

## 11. Derived Cognition Is Fallible

Dynamic Self, Relationship State, Reflection output, and future internal processes are interpretations.

They are not ground truth.

Observed experience has higher evidential authority than derived cognition.

Derived state must not become stronger merely because it caused APERTURE to repeat itself later.

## 12. Experimental Feature Policy

Experimental systems are evaluated by ablation.

For each feature:

1. run APERTURE without it
2. run APERTURE with it
3. compare meaningful behavior
4. retain it only if it provides a clear benefit

The question is not:

> Can we make this subsystem technically work?

The question is:

> Does APERTURE become meaningfully better because this subsystem exists?

## 13. Development Priority

Development order:

1. stable baseline
2. useful long-term continuity
3. reliable interaction with tools and workspace
4. evaluation of existing experiments
5. only then new cognition features
6. broader capabilities
7. voice and polish

Identity research remains important, but it does not block practical agent development.

## 14. Stop Rule

When a feature works well enough for its current purpose, move forward.

Do not keep refining it solely because another synthetic edge case can be invented.

Return to an older subsystem only when:

* real usage exposes a meaningful failure
* another feature depends on it
* or an intentional evaluation shows that it is limiting APERTURE

## 15. Definition of APERTURE v0.1

APERTURE v0.1 does not require a sophisticated personality.

A successful baseline can:

* run locally
* maintain conversation
* use tools
* remember across sessions
* preserve memory ownership
* retrieve useful prior information
* record what actually happened
* preserve changes over time
* distinguish observation from derived interpretation
* act consistently enough that previous interactions matter later

If those properties work, APERTURE already exists as an agent.

Everything beyond them is an experiment built on top of that foundation.

## 16. Final Constraint

The project should always be able to answer:

> What does this component allow APERTURE to do that it could not meaningfully do before?

If there is no clear answer, do not expand the architecture.
