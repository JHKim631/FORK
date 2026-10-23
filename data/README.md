# Data

The prepared training and evaluation splits and the two out-of-domain pools
the paper uses, as committed files. Every set was built by the builders in
`src/fork/data/` from a pinned source; `bash scripts/prepare_data.sh --rebuild`
rebuilds all of them in place, and the result is byte-identical.

| set | files | rows | built by |
|---|---|---|---|
| RubricHub-Medical | `medical/train.parquet`, `medical/eval.parquet`, `medical/meta.json` | 12,519 / 1,391 | `configs/data/medical.yaml` |
| RubricHub-Science, curated | `science/train.parquet`, `science/eval.parquet`, `science/meta.json`, `science/sources.json` | 12,519 / 1,467 | `configs/data/science_raw.yaml`, then `configs/data/science.yaml` |
| HealthBench pool | `healthbench/eval.parquet`, `healthbench/meta.json` | 100 conversations, 1,210 rubric items | `python -m fork.data.ood --bench healthbench` |
| ResearchQA pool | `researchqa/eval.parquet`, `researchqa/meta.json` | 100 queries, 764 rubric items | `python -m fork.data.ood --bench researchqa` |

`scripts/prepare_data.sh` (without `--rebuild`) only converts the two splits
to verl's dataset format under `data/<domain>/verl/`, which is not committed.

## Format

### The prepared splits (`medical/`, `science/`)

One row per prompt. The `eval.parquet` of a domain is its test pool; the
evaluation draws its subsets from it with the seeds in `configs/eval/<domain>.yaml`.

| column | type | description |
|---|---|---|
| `prompt_id` | `string` | `med-` or `sci-` plus the first 12 hex digits of the SHA-1 of the prompt text. Stable across rebuilds; the GRPO group key during training. |
| `prompt` | `string` | The user prompt, exactly what the policy sees (no system message). |
| `rubrics` | `list<struct<text: string, weight: double>>` | The rubric items and their weights (RubricHub's `criterion` and `points`; the weights are positive integers 1–20). |
| `reference_answer` | `string` | Empty: RubricHub carries no reference answers. |
| `source` | `string` | `rubrichub_medical` or `rubrichub_science`. |

`meta.json` records the source file's SHA-256, the filter counts and the
distributions of rubric counts and prompt lengths. `science/sources.json` maps
every prompt of the full science slice to the dataset(s) it was attributed to.

### The OOD pools (`healthbench/`, `researchqa/`)

The same `prompt_id` / `prompt` / `rubrics` / `source` columns, so the
generation and the rubric-free judge read a pool like an in-domain split, plus
what each benchmark's own grader needs:

- HealthBench: `messages` holds the conversation turns as a JSON list (every
  conversation ends with a user turn; `prompt` is the turns rendered
  `role: content` and joined by blank lines); each rubric item carries its
  `points` as `weight` (negative for undesirable content) and its `tags`.
- ResearchQA: each rubric item has `weight` 1.0 (the benchmark is unweighted)
  and its `types`; the rows keep `general_domain`, `subdomain` and `field`.

## Provenance

**RubricHub-Medical** — `sojuL/RubricHub_v1`, file `RuRL/rurbichub_v1_Medical.parquet`,
commit `3837d55971473a872e84879c88f708b8da3ec2ef` (Apache-2.0). 29,681 rows; 4
dropped for a rubric count outside [8, 50]; 29,677 usable; a seeded shuffle
(seed 20260707) splits off 1,391 eval and 12,519 train prompts.

**RubricHub-Science** — the same dataset, file `RuRL/rurbichub_v1_Science.parquet`.
29,418 rows; 3 dropped for the rubric count, 60 for a prompt above the policy's
584-token budget (the chat-templated length under the Qwen2.5-7B-Instruct
tokenizer); 29,355 usable; the seeded split gives 2,201 eval and 19,806 train
prompts (`data/science_raw`, not committed). The curation then attributes every
one of those prompts to the datasets RubricHub drew them from — RaR-Science
(`ScaleAI/RaR-Science`), MegaScience (`MegaScience/MegaScience`) and ResearchQA
(`realliyifei/ResearchQA`), each at a pinned revision — by exact text match,
and keeps the RaR-Science and MegaScience ones: 13,096 of the train prompts, of
which 12,519 are drawn with the seed (the medical train count, so both domains
train for 291 steps), and all 1,467 of the eval prompts. ResearchQA-sourced
prompts are excluded because ResearchQA is the science arms' out-of-domain
benchmark. The kept rows are copied unchanged.

**HealthBench** — the public `oss_eval` file of OpenAI simple-evals
(`2025-05-07-06-14-12_oss_eval.jsonl`, SHA-256
`e99dd3c6…ecd2f`, 5,000 conversations; MIT). The pool is the official draw:
`random.Random(0).sample(examples, 100)` over the file's order.

**ResearchQA** — `realliyifei/ResearchQA`, the test split
(`data/test-00000-of-00001.parquet` at commit `bf8a4cfef073ecfc0275c57acf8ca960e4dc79d6`,
3,750 queries; MIT), restricted to the two natural-science domains
(Life & Earth Sciences, Physical & Theoretical Sciences: 1,100 queries),
sorted by id, then `random.Random(0).sample(..., 100)`.
