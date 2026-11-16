# Development-only validation prompt comparison

Date: 2026-11-16. This experiment tested whether a small clarification in the reflection validator prompt would reduce **false active user-memory writes**. It used only the ten synthetic development cases in `evaluations/memory_acceptance/cases.json`, temporary stores, and local Ollama. No held-out or private conversation cases were used for prompt selection.

| Model | Prompt | Exact decisions | False active writes | Create | Pending | Supersede |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qwen3:8b` | Existing | 8/10 | 2 | 3/3 | 0/1 | 1/1 |
| `qwen3:8b` | General decision order | 8/10 | 2 | 3/3 | 0/1 | 1/1 |
| `qwen3:8b` | Decision order plus unrelated-topic examples | 8/10 | 2 | 3/3 | 0/1 | 1/1 |
| `qwen3.5:4b` | Existing | 8/10 | 2 | 3/3 | 0/1 | 1/1 |
| `qwen3.5:4b` | General decision order | 9/10 | 1 | 3/3 | 1/1 | 1/1 |

The `qwen3:8b` rows were run on the same working tree, fixture, and model digest, with only the prompt block varied. Each run completed ten cases, attempted 20 model calls, and reported zero model errors and provenance failures. The `qwen3.5:4b` baseline was recorded on the same working tree before this experiment. Both model variants still produced an incorrect active memory from a hypothetical alternative to a current preference. `qwen3:8b` also saved an unsettled routine actively despite the clarification and examples.

The prompt variants were **reverted** because neither met the development acceptance condition of zero false active writes while preserving positive capture on the committed default model. No runtime code or prompt change from this experiment was retained. These ten cases are development data; their scores are diagnostic, not an unbiased estimate of generalization. A future design should keep semantic interpretation in model passes, preserve the structural provenance and lineage checks, and use a newly frozen held-out set after any new change. Do not reinterpret the prior 20-case held-out result as fresh evidence after tuning.

Local raw reports are ignored by Git under `.reliability-tmp/`: `memory-acceptance-dev-baseline-8b-same-tree.json`, `memory-acceptance-dev-decision-order-8b.json`, `memory-acceptance-dev-examples-8b.json`, and `memory-acceptance-dev-decision-order.json`. Each report contains the fixture and source hashes needed to check the comparison.
