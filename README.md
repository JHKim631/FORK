<!--
  Placeholders still open (search for "TODO(identity)"):
    - the arXiv id in the badge, the paper link and the BibTeX (XXXX.XXXXX)
    - the dates in "Updates"
    - assets/figure2.png (Figure 2 of the paper)
-->
# Eat Your Precision: Precision Objective Mitigates Reward Hacking in Rubric-Based RL

[![arXiv](https://img.shields.io/badge/Paper-arXiv:XXXX.XXXXX-Green)](https://arxiv.org/abs/XXXX.XXXXX)  [![BibTex](https://img.shields.io/badge/Paper-BibTex-yellow)](#bibtex)  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Official implementation of **FORK**, a reward for rubric-based reinforcement learning that jointly optimizes rubric recall and a unit-level precision objective, as presented in our paper:  
[**Eat Your Precision: Precision Objective Mitigates Reward Hacking in Rubric-Based RL**](https://arxiv.org/abs/XXXX.XXXXX)  
by Hyun Ryu<sup>\*1</sup>, Jong-Heon Kim<sup>\*1</sup>, Yiqing Xie<sup>2</sup>, Seungone Kim<sup>2</sup>, Carolyn Rosé<sup>3</sup> and Eunho Yang<sup>1</sup>  
<sup>1</sup>Korea Advanced Institute of Science & Technology &nbsp;&nbsp; <sup>2</sup>Carnegie Mellon University &nbsp;&nbsp; <sup>3</sup>School of Computer Science, Carnegie Mellon University &nbsp;&nbsp; <sup>\*</sup>Equal Contribution

<p align="center">
  <img src="assets/figure2.png" alt="Training with FORK" width="90%" />  <!-- TODO(identity): Figure 2 of the paper -->
</p>

---

## 🔔 Updates
- [✔] (YY.MM.DD) Code and data are out.  <!-- TODO(identity) -->
- [✔] (YY.MM.DD) Paper is out! [here](https://arxiv.org/abs/XXXX.XXXXX)  <!-- TODO(identity) -->

---

## 📖 Overview

Rubric-based rewards score a response by the weighted coverage of a prompt's rubrics, so a policy trained against them maximizes a recall objective: responses become more complete, but also longer and full of unnecessary or incorrect content (paper Figure 1). FORK complements rubric recall with a **precision** objective over the units of a response (Section 3):

- **Response unitization.** The response is split into units of about one sentence each (`fork.rewards.unitize`).
- **Credit-based necessity.** A grader matches every rubric to the units that address it (*Rubrics to Units Matching*, Figure 5); each rubric's unit credit of one is shared equally among its matched units, so redundant units earn only a fraction of it, and a unit's credit is the sum over the rubrics it serves. The grader then sets the threshold a unit's credit must reach to be necessary (*Necessity Threshold Decision*, Figures 6–7).
- **Factual correctness.** Independently, the grader labels every unit factually correct or not (*Factual Correctness Decision*, Figures 8–9).
- **Precision** is the fraction of units that are both necessary and factually correct; **recall** is the standard weighted rubric coverage; the reward is their **F<sub>β</sub>** (weighted harmonic mean, Eq. 6), with β = 1 or β = 1.25.

The repository contains the reward as a [verl](https://github.com/volcengine/verl) custom reward function, the training grader (Qwen3.6-35B-A3B served by vLLM), the prepared training data, and the full evaluation of the paper: rubric-based recall / precision / F1, rubric-free win rates, HealthBench and ResearchQA, the presence- / absence-based rubric analysis and the training trajectory, with GPT-5.4 as the reference grader or the training grader as the judge. Three reward formulations of the paper's main comparison can be trained:

| method | reward | precision |
|---|---|---|
| **Standard** rubric-coverage reward (Gunjal et al., 2026) | recall | — |
| **Generic Criteria** (paper Section 4.1, Appendix A; Figure 14) | F<sub>1</sub>(precision, recall) | necessity and factual correctness of every unit, labeled directly by one grader call |
| **FORK** (ours) | F<sub>β</sub>(precision, recall) | the credit-based necessity above, and the factual-correctness grader |

The other baselines and ablations of the paper are not part of this release.

## 🔧 Environment Setup

The paper's runs use 8 NVIDIA H100 (80 GB): the policy trainer on four GPUs and four replicas of the training grader on the other four. Two Python environments are needed, because the grader model requires a newer vLLM than verl 0.7.0 allows.

1. The **trainer** environment — training, data preparation and evaluation (verl 0.7.0, vLLM 0.12, PyTorch 2.9, flash-attn 2.8 and this package; the exact versions of the paper's runs are pinned):
```bash
conda create -n fork python=3.10 && conda activate fork
pip install -r requirements/trainer.txt
```

2. The **grader** environment — only `scripts/serve_grader.sh` runs in it (vLLM 0.26, Transformers 5.14):
```bash
conda create -n fork-grader python=3.12 && conda activate fork-grader
pip install -r requirements/grader.txt
```

3. Keys. Copy `.env.example` to `.env` and fill in what you use:
```bash
cp .env.example .env
```
`OPENAI_API_KEY` is needed by the evaluation (GPT-5.4 as the reference grader); `WANDB_API_KEY` is optional (without it, training logs to the console only); `HF_TOKEN` is only needed for gated downloads.

> Without a GPU, `pip install -e .` installs what the reward function, the data builders and the evaluation report import, which is enough to read the code, rebuild the data and aggregate evaluation lanes.

## 🏋️ Data

The prepared splits and the two out-of-domain pools of the paper are committed under [`data/`](./data); see [`data/README.md`](./data/README.md) for the formats and the provenance of each file.

| set | files | rows |
|---|---|---|
| RubricHub-Medical | `data/medical/{train,eval}.parquet` | 12,519 train / 1,391 test |
| RubricHub-Science (ResearchQA-sourced prompts excluded) | `data/science/{train,eval}.parquet` | 12,519 train / 1,467 test |
| HealthBench pool | `data/healthbench/eval.parquet` | 100 conversations, 1,210 rubric items |
| ResearchQA pool | `data/researchqa/eval.parquet` | 100 queries, 764 rubric items |

Before training, convert the splits to verl's dataset format (seconds, no network):
```bash
bash scripts/prepare_data.sh
```

Every file was built by the builders in `src/fork/data/` from a pinned source (a RubricHub commit, the public HealthBench file by SHA-256, a ResearchQA revision); `bash scripts/prepare_data.sh --rebuild` downloads the sources and rebuilds all of them byte for byte.

## 🚀 Training

**1. Serve the training grader** (in the grader environment; GPUs 4–7 by default). Leave it running:
```bash
bash scripts/serve_grader.sh
```
The script starts four vLLM replicas of `Qwen/Qwen3.6-35B-A3B-FP8` on ports 8001–8004 (`configs/grader/qwen36_35b.yaml`), waits for them to be healthy and prints their URLs. On a 4-GPU node: `REPLICAS=2 GPU_OFFSET=2 bash scripts/serve_grader.sh`.

**2. Train** (in the trainer environment; GPUs 0–3 by default):
```bash
bash scripts/train.sh configs/train/fork_beta1.25_medical.yaml
```

One config per run of the paper's main tables:

| config | method | domain |
|---|---|---|
| `configs/train/standard_medical.yaml` | Standard | RubricHub-Medical |
| `configs/train/generic_criteria_medical.yaml` | Generic Criteria (β = 1) | RubricHub-Medical |
| `configs/train/fork_beta1_medical.yaml` | FORK, β = 1 | RubricHub-Medical |
| `configs/train/fork_beta1.25_medical.yaml` | FORK, β = 1.25 | RubricHub-Medical |
| `configs/train/standard_science.yaml` | Standard | RubricHub-Science |
| `configs/train/fork_beta1.25_science.yaml` | FORK, β = 1.25 | RubricHub-Science |

A run config holds only the reward's switches (`reward: recall | fbeta`, `beta`, `precision: credit | generic`, `domain: medical | science`); the GRPO hyperparameters shared by every run are in [`configs/grpo.yaml`](./configs/grpo.yaml) (16 rollouts per prompt, 128 prompts per step, learning rate 4.2e-6 with a 5 % warmup, 3 epochs = 291 steps, a checkpoint every 25 steps). `bash scripts/train.sh <config> --print` prints the verl command line without launching anything.

A run writes to `outputs/train/<run_id>/`: the checkpoints under `ckpts/global_step_<n>/`, every grader call of every rollout under `grader_io/step_<n>.jsonl`, the per-step reward statistics in `reward_log.jsonl`, and `manifest.json`. Rerunning the same command resumes from the last checkpoint. Knobs are environment variables documented at the top of [`scripts/train.sh`](./scripts/train.sh): `TRAIN_GPUS`, `GRADER_URLS`, `MICRO_BSZ`, `OPTIM_OFFLOAD`, `TOTAL_STEPS`, `WANDB_PROJECT`, …

## 📊 Evaluation

`scripts/eval.sh` evaluates one model the way the paper's Tables 1 and 2 do: it merges the checkpoint, generates greedy responses on the 100-prompt test subset and on the out-of-domain pool, grades rubric recall and precision, judges the responses pairwise against the base policy, grades the out-of-domain pool the benchmark's own way, and prints the row in the paper's columns. The base policy is the comparator, so evaluate it first, once per domain:

```bash
bash scripts/eval.sh base_medical                        # the untrained policy
bash scripts/eval.sh fork_beta1.25_medical               # a run, at its last checkpoint
bash scripts/eval.sh fork_beta1.25_medical --step 250    # a specific checkpoint
```

Every step is guarded by its output file, so rerunning the same command resumes an interrupted evaluation. The judge is GPT-5.4 through the OpenAI API (`--judge gpt54`, the default; `OPENAI_API_KEY` in `.env`) or the training grader on the served replicas (`--judge local`, the setting of the paper's Figure 3). Generation uses one GPU (`GEN_GPU`, default 0).

Results go to `outputs/eval/<label>/` with `<label>` = `<run_id>@<step>` or `base_<domain>`:

```
outputs/eval/fork_beta1.25_medical@291/
├── grader_responses.jsonl           the responses on the test subset
├── ood_responses.jsonl              the responses on the HealthBench pool
├── rubric_based.gpt54/              recall_scored.jsonl, precision_scored.jsonl, summary.json
├── rubric_free.gpt54.p1/            vs_base_medical.jsonl  (the five-dimension Likert judge, both orders)
├── rubric_free.gpt54.p2/            vs_base_medical.jsonl  (the MT-Bench judge prompt)
├── ood.gpt54/                       graded.jsonl, summary.json  (HealthBench's own grading)
└── ood_rubric_free.gpt54.p1/        vs_base_medical.jsonl
```

`python -m fork.eval.report` aggregates the lanes of any labels into the paper's columns (macro recall, macro precision, the per-response F1 mean, the win rate with ties excluded and position-flip averaging, the mean response length, and the out-of-domain recall and win rate):

```bash
python -m fork.eval.report --label base_medical --label standard_medical@291 --label fork_beta1.25_medical@291 --vs base_medical
python -m fork.eval.report --label fork_beta1.25_medical@291 --vs base_medical --dimensions   # the five quality dimensions (Table 8)
python -m fork.eval.report --label fork_beta1.25_medical@291 --vs base_medical --prompt p2     # the second judge prompt (Table 4)
```

**The training trajectory** (Figure 1) scores every checkpoint of a run on the 300-prompt trajectory subset with the training grader as the judge — recall, precision and the win rate against the base policy — and needs the grader replicas and one generation GPU:
```bash
bash scripts/trajectory.sh fork_beta1.25_medical
```

**Presence- and absence-based rubrics** (Table 3): one pass types every rubric item of the evaluation subsets with the training grader, and the report then pools the per-item verdicts of the rubric-based lanes by type:
```bash
python -m fork.eval.rubric_types --config configs/eval/medical.yaml          # -> outputs/rubric_types/medical.jsonl
python -m fork.eval.report --label fork_beta1.25_medical@291 --rubric-types outputs/rubric_types/medical.jsonl
```

The evaluation settings of a domain — the test pool, the subsets and their seeds, the generation settings, the two judges, the judge prompts and the out-of-domain benchmark — are in [`configs/eval/medical.yaml`](./configs/eval/medical.yaml) and [`configs/eval/science.yaml`](./configs/eval/science.yaml). The scripts call the modules under `src/fork/eval/`, which can be run on their own (`python -m fork.eval.<module> --help`): `generate`, `rubric_based`, `rubric_free`, `ood`, `rubric_types`, `report`.

## 🔬 Reproducing the paper

| paper | commands |
|---|---|
| Table 1 (medical) | train the four medical configs; `bash scripts/eval.sh base_medical`, then `bash scripts/eval.sh <run_id>` for each run; `python -m fork.eval.report --label … --vs base_medical` |
| Table 2 (science) | the same with `standard_science`, `fork_beta1.25_science` and `base_science` |
| Table 3 (presence- / absence-based rubric satisfaction) | `python -m fork.eval.rubric_types --config configs/eval/medical.yaml`, then the report with `--rubric-types` |
| Table 4 (the second judge prompt) | the report with `--prompt p2` |
| Table 8 (the five quality dimensions) | the report with `--dimensions` |
| Figure 1 (the training trajectory) | `bash scripts/trajectory.sh <run_id>` |
| Figure 3 (evaluation with the training grader) | `bash scripts/eval.sh <run_id> --judge local` |
| Figures 5–9 and 14 (the training prompts) | the files under `prompts/train/` |

## 📝 Prompts

Every prompt used by the training rewards and the evaluation lives under [`prompts/`](./prompts) as a text file with a `===SYSTEM===` and a `===USER===` section. The body of each file is, byte for byte, the prompt the paper's runs were graded with; the header comment names its role in the paper and its provenance.

| file | role |
|---|---|
| `train/coverage_grading.txt` | rubric recall — the standard rubric-coverage reward (Section 3.1); the grading prompt of Mahmoud et al. (2026) |
| `train/fork_match_rubrics.txt` | FORK, Rubrics to Units Matching (Figure 5) |
| `train/fork_necessity_threshold.txt` | FORK, Necessity Threshold Decision (Figures 6–7) |
| `train/fork_correctness.txt` | FORK, Factual Correctness Decision (Figures 8–9) |
| `train/*_science.txt` | the science-domain wording of the three FORK prompts |
| `train/generic_criteria.txt` | Generic Criteria (Figure 14); `generic_criteria_audit.txt` is its refusal-rescue twin, sent once when the grader refuses a call |
| `eval/precision_label.txt` | the precision grader of the rubric-based evaluation (Appendix E.2), with its science twin and its audit twin |
| `eval/rubric_free_judge.txt` | the rubric-free pairwise judge (the judge prompt of Mahmoud et al., 2026), with its science twin |
| `eval/rubric_free_judge_mtbench.txt` | the second judge prompt (Table 4): MT-Bench "pair-v2" (Zheng et al., 2023) |
| `eval/healthbench_grader.txt` | HealthBench's own grader prompt (OpenAI simple-evals) |
| `eval/researchqa_coverage.txt` | ResearchQA's own rubric-coverage judge prompt |
| `eval/rubric_types.txt` | the presence- / absence-based typing of rubric items (Table 3) |

## 🗂️ Repository layout

```
configs/
  grpo.yaml                 the GRPO hyperparameters shared by every run
  train/<run_id>.yaml       one run of the paper's main tables: the reward's switches
  grader/qwen36_35b.yaml    the training grader: serving geometry and the call parameters
  data/*.yaml               the data builders' sources (pinned revisions) and splits
  eval/<domain>.yaml        the evaluation of a domain: pool, subsets, judges, judge prompts, OOD benchmark
data/                       the prepared splits and the two OOD pools (data/README.md)
prompts/train, prompts/eval the grader and judge prompts
scripts/
  prepare_data.sh           the shipped data -> verl's format (--rebuild: everything from the sources)
  serve_grader.sh           the training grader's vLLM replicas
  train.sh                  one GRPO run (verl), resumable
  eval.sh                   one model's evaluation (Tables 1 and 2)
  trajectory.sh             every checkpoint of a run with the training grader (Figure 1)
src/fork/
  grader/                   the grader client: the calls, the prompt templates, the parsers
  rewards/                  recall, unitization, the FORK precision chain, Generic Criteria, F-beta,
                            the group-median fallback, and verl_reward.py (verl's custom reward function)
  data/                     the data builders
  eval/                     the evaluation lanes and the report
  launch.py                 the run config's validation and verl's command line (scripts/train.sh)
requirements/               the trainer and grader environments, pinned
```

<a id="bibtex"></a>
## 📚 BibTeX
If you find this repository useful for your research, please consider citing us:

```
@article{ryu2026eat,
  title={Eat Your Precision: Precision Objective Mitigates Reward Hacking in Rubric-Based RL},
  author={Ryu, Hyun and Kim, Jong-Heon and Xie, Yiqing and Kim, Seungone and Rose, Carolyn and Yang, Eunho},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
<!-- TODO(identity): the arXiv id -->

## 📄 License and acknowledgements

This repository is released under the [Apache License 2.0](LICENSE).

Training runs on [verl](https://github.com/volcengine/verl); the training data comes from [RubricHub](https://huggingface.co/datasets/sojuL/RubricHub_v1), whose science prompts are attributed to [RaR-Science](https://huggingface.co/datasets/ScaleAI/RaR-Science), [MegaScience](https://huggingface.co/datasets/MegaScience/MegaScience) and [ResearchQA](https://huggingface.co/datasets/realliyifei/ResearchQA); the out-of-domain evaluation uses [HealthBench](https://github.com/openai/simple-evals) and [ResearchQA](https://github.com/realliyifei/ResearchQA) with their own grader prompts; the rubric recall prompt and the rubric-free judge prompt follow Mahmoud et al. (2026), and the second judge prompt is MT-Bench's (Zheng et al., 2023). The training grader is [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) and the base policy is [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

## ✉️ Contact
If you have any questions or feedback, feel free to reach out:
- Hyun Ryu: ryuhyun1905@kaist.ac.kr
- Jong-Heon Kim: jongheon@kaist.ac.kr
