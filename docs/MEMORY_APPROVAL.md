# Pending user memory approval

Automatic reflection queues user memory candidates for review by default. A
pending proposal becomes available to memory search and conversation context
only after explicit approval. Pending replacements leave the previous memory
active.

Set `$env:APERTURE_REQUIRE_REFLECTION_USER_MEMORY_APPROVAL = '0'` before
starting APERTURE to allow automatic active writes for candidates the model
marks as clearly supported. Candidates the model flags for confirmation still
go to the queue. See the [synthetic policy comparison](MEMORY_ACCEPTANCE_EVALUATION.md#november-2026-policy-comparison).

## CLI commands

- `/memory list` shows pending proposals, their original user messages, and any memory they would replace.
- `/memory show <id>` shows one proposal in full.
- `/memory approve <id>` creates the approved user memory, or applies its proposed temporal replacement.
- `/memory reject <id>` closes the proposal without changing active memory.

The commands are interpreted locally before the model receives a turn. Text found inside a proposal or source message cannot execute a command. Startup announces pending proposals but never approves them automatically.

## Authority and provenance

Each proposal stores the original observed user message IDs and episode ID. Approval checks those historical events again, so a decision can be made after a restart without accepting an assistant message or a derived event as user evidence. An approval or rejection command is recorded as a separate observed `user_command` event and linked to the decision. Command events are not eligible sources for automatic reflection.

Approval checks that a proposed replacement still targets the same active user memory version shown when the proposal was created. A stale target leaves the proposal pending for review. The proposal decision and memory mutation are committed in one transaction in the memory database. The experience event is stored separately; it records the user's decision attempt, while the memory database holds the authoritative outcome.

The model decides whether a candidate is clearly supported, uncertain, or unsuitable. Code enforces the candidate shape, observed source ownership, current target version, and explicit decision boundary. These structural checks do not establish semantic truth; conversation evaluations should continue to test the model's judgments.
