# Synthetic memory acceptance baseline

This baseline measures the reflection-to-memory path in the committed APERTURE snapshot `3d552b8` with local Ollama model `qwen3:8b` (model digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`). It was run from a clean archive of that commit on Windows. The fixture SHA-256 is `84aa14c8000bab47f7ad8068d3c037fe8b35c9505e9c9b397dc010faada35e28`. Both splits used the same source hashes; the generated JSON reports retain the full hash map, model settings, raw responses, event provenance, and memory state.

| Split | Cases | Exact structural decisions | Create | Reject | Pending | Supersede | False active writes | Model errors | Provenance failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 10 | 8 | 3/3 | 4/5 | 0/1 | 1/1 | 2 | 0 | 0 |
| Held out | 20 | 16 | 7/7 | 7/9 | 0/2 | 2/2 | 4 | 0 | 0 |

The held-out run attempted 40 model calls across 20 cases. Its four decision errors were two uncertain statements saved as active memories and two questions or conditional statements treated as updates to existing preferences. The development run had one uncertain statement saved actively and one hypothetical statement treated as a supersession. These are material memory safety failures; a correct rejection rate alone would hide them.

The score checks the case-level decision, user-event provenance, ownership, and supersession lineage. **Semantic fidelity has not been reviewed by a human** (`semantic_reviewed: 0` in both reports), so 24/30 structural matches must not be presented as 24 semantically correct memories. The cases are synthetic and the held-out labels are visible in the repository. They do not estimate performance on the user's private conversations or prove that actor activity claims correspond to completed tool actions.

## Reproduction

In a clean checkout of `3d552b8` with the project virtual environment and local `qwen3:8b` model installed, run from the repository root:

```powershell
.\.venv\Scripts\python.exe -B tools\memory_acceptance_eval.py --live --split dev --max-cases 10 --max-calls 22 --timeout-seconds 60 --max-seconds 600 --output .reliability-tmp\memory-acceptance-dev.json
.\.venv\Scripts\python.exe -B tools\memory_acceptance_eval.py --live --split holdout --max-cases 20 --max-calls 44 --timeout-seconds 60 --max-seconds 900 --output .reliability-tmp\memory-acceptance-holdout.json
```

The report files are local and ignored by Git. A rerun needs a new output filename or the deliberate `--overwrite` flag. See [MEMORY_ACCEPTANCE_EVALUATION.md](MEMORY_ACCEPTANCE_EVALUATION.md) for interpretation and timeout semantics.

## Verification at the measured commit

- Clean archive dry run: 30 cases prepared, zero model calls.
- Isolated evaluator tests: 11/11 passed; reflection regression tests: 12/12 passed.
- Full isolated suite: 175/178 passed, with no guard violations. The three failures were Windows permission errors when tests attempted hard-link or junction creation (`test_mode_readonly_hardlinks_and_worst_escaping`, `test_windows_junctions_are_never_followed`, `test_write_junction_boundary`). These checks did not establish a product regression; they remain unverified in this environment.
