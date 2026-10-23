<!--
  Placeholders still open (search for "TODO(identity)"):
    - the arXiv id in the badge, the paper link and the BibTeX (XXXX.XXXXX)
    - the dates in "Updates"
-->
# Eat Your Precision: Precision Objective Mitigates Reward Hacking in Rubric-Based RL

[![arXiv](https://img.shields.io/badge/Paper-arXiv:XXXX.XXXXX-Green)](https://arxiv.org/abs/XXXX.XXXXX)  [![BibTex](https://img.shields.io/badge/Paper-BibTex-yellow)](#bibtex)  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Official implementation of **FORK**, a reward for rubric-based reinforcement learning that jointly optimizes rubric recall and a unit-level precision objective, as presented in our paper:  
[**Eat Your Precision: Precision Objective Mitigates Reward Hacking in Rubric-Based RL**](https://arxiv.org/abs/XXXX.XXXXX)  
by Hyun Ryu<sup>\*1</sup>, Jong-Heon Kim<sup>\*1</sup>, Yiqing Xie<sup>2</sup>, Seungone Kim<sup>2</sup>, Carolyn Rosé<sup>3</sup> and Eunho Yang<sup>1</sup>  
<sup>1</sup>Korea Advanced Institute of Science & Technology &nbsp;&nbsp; <sup>2</sup>Carnegie Mellon University &nbsp;&nbsp; <sup>3</sup>School of Computer Science, Carnegie Mellon University &nbsp;&nbsp; <sup>\*</sup>Equal Contribution

<p align="center">
  <img src="assets/fork_overview.png" alt="Training with FORK" width="90%" />
</p>

---

## 🔔 Updates
- [✔] (YY.MM.DD) Code and data are out.  <!-- TODO(identity) -->
- [✔] (YY.MM.DD) Paper is out! [here](https://arxiv.org/abs/XXXX.XXXXX)  <!-- TODO(identity) -->

---

## 📖 Overview

Rubric-based rewards score a response by its weighted rubric coverage, a recall objective, so a policy trained against them learns to be complete but also verbose and inaccurate. FORK adds a precision objective (paper Section 3): the response is split into sentence-level units; a grader matches rubrics to the units that address them and shares each rubric's credit among them, then sets the credit threshold a unit must reach to be necessary (Figures 5–7); a second call labels every unit factually correct or not (Figures 8–9). Precision is the fraction of units that are both necessary and correct, recall is the standard rubric coverage, and the reward is their F<sub>β</sub> (Eq. 6).

This repository contains:

- the reward as a [verl](https://github.com/volcengine/verl) custom reward function (`src/fork/rewards/`, `src/fork/grader/`), in the three formulations of the paper's main comparison: **Standard** (recall only), **Generic Criteria** (precision from one labeling call) and **FORK** (credit-based necessity and factual correctness, β = 1 or 1.25);
- the training grader, Qwen3.6-35B-A3B served by vLLM (`configs/grader/`, `scripts/serve_grader.sh`);
- the prepared RubricHub-Medical and RubricHub-Science splits and the HealthBench and ResearchQA pools, with the builders that rebuild them from pinned sources (`data/`, `src/fork/data/`);
- the evaluation behind the paper's tables and figures, with GPT-5.4 or the training grader as the judge (`src/fork/eval/`, `scripts/eval.sh`, `scripts/trajectory.sh`).

The other baselines and ablations of the paper are not part of this release.

## 🔧 Setup

The paper's runs use 8 NVIDIA H100: the policy trainer on four GPUs and four replicas of the training grader on the other four. Two environments are needed, because the grader model requires a newer vLLM than verl 0.7.0 allows:

```bash
conda create -n fork python=3.10 && conda activate fork
pip install -r requirements/trainer.txt          # training, data preparation and evaluation (verl 0.7.0, vLLM 0.12, PyTorch 2.9)
conda create -n fork-grader python=3.12 && conda activate fork-grader
pip install -r requirements/grader.txt           # serving the training grader only (vLLM 0.26)
cp .env.example .env                             # OPENAI_API_KEY for the evaluation judge; WANDB_API_KEY and HF_TOKEN are optional
```

## 🏋️ Data

[`data/`](./data) ships the prepared splits — RubricHub-Medical (12,519 train / 1,391 test) and RubricHub-Science with its ResearchQA-sourced prompts excluded (12,519 / 1,467) — and the two out-of-domain pools (100 HealthBench conversations, 100 ResearchQA queries). Formats and provenance are in [`data/README.md`](./data/README.md). Before training, convert the splits to verl's format:

```bash
bash scripts/prepare_data.sh        # --rebuild: everything again from the pinned sources, byte for byte
```

## 🚀 Training

```bash
bash scripts/serve_grader.sh                                      # grader environment; GPUs 4-7, leave it running
bash scripts/train.sh configs/train/fork_beta1.25_medical.yaml    # trainer environment; GPUs 0-3
```

[`configs/train/`](./configs/train) holds one config per run of the paper's main tables — `standard_medical`, `standard_science`, `generic_criteria_medical`, `fork_beta1_medical`, `fork_beta1.25_medical`, `fork_beta1.25_science` — each only the reward's switches (`reward`, `beta`, `precision`, `domain`); the GRPO hyperparameters shared by every run are in [`configs/grpo.yaml`](./configs/grpo.yaml). A run writes its checkpoints, every grader call of every rollout and the per-step reward statistics to `outputs/train/<run_id>/`, and rerunning the same command resumes it. `--print` shows the verl command line; the GPU and batch knobs are documented at the top of [`scripts/train.sh`](./scripts/train.sh).

## 📊 Evaluation

```bash
bash scripts/eval.sh base_medical                      # the base policy first: it is the comparator
bash scripts/eval.sh fork_beta1.25_medical             # a run at its last checkpoint (--step N for another)
python -m fork.eval.report --label base_medical --label fork_beta1.25_medical@291 --vs base_medical
```

`eval.sh` merges the checkpoint, generates greedy responses on the 100-prompt test subset and on the out-of-domain pool, grades rubric recall and precision, judges the responses pairwise against the base policy under two judge prompts, grades the out-of-domain pool the benchmark's own way, and prints the row in the paper's columns (Tables 1 and 2); every step is guarded by its output file, so rerunning resumes. The judge is GPT-5.4 (`--judge gpt54`, the default) or the training grader on its replicas (`--judge local`, Figure 3). The report also takes `--prompt p2` (Table 4), `--dimensions` (Table 8) and `--rubric-types` (Table 3, after `python -m fork.eval.rubric_types --config configs/eval/medical.yaml`); `bash scripts/trajectory.sh <run_id>` scores every checkpoint of a run with the training grader (Figure 1). The settings of each domain are in [`configs/eval/`](./configs/eval).

## 📝 Prompts

Every grader and judge prompt is a text file under [`prompts/`](./prompts): the three FORK calls (Figures 5–9), Generic Criteria (Figure 14), the rubric recall and rubric-free judge prompts of Mahmoud et al. (2026), the MT-Bench judge prompt (Table 4), the two benchmarks' own graders and the rubric typing prompt (Table 3). The body of each file is, byte for byte, the prompt the paper's runs used; the header names its role and provenance.

## 🗂️ Repository layout

```
configs/        grpo.yaml (shared hyperparameters), train/ (the runs), grader/ (the training grader), data/ (the builders), eval/ (per domain)
data/           the prepared splits and the OOD pools (data/README.md)
prompts/        train/ (the reward graders), eval/ (the judges)
scripts/        prepare_data.sh, serve_grader.sh, train.sh, eval.sh, trajectory.sh
src/fork/       grader/ (client, templates, parsers), rewards/ (recall, unitization, the FORK chain, Generic Criteria, F-beta, verl_reward.py),
                data/ (the builders), eval/ (the lanes and the report), launch.py (the run config and verl's command line)
requirements/   the two pinned environments
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

Released under the [Apache License 2.0](LICENSE). Training runs on [verl](https://github.com/volcengine/verl); the data comes from [RubricHub](https://huggingface.co/datasets/sojuL/RubricHub_v1), whose science prompts are attributed to [RaR-Science](https://huggingface.co/datasets/ScaleAI/RaR-Science), [MegaScience](https://huggingface.co/datasets/MegaScience/MegaScience) and [ResearchQA](https://huggingface.co/datasets/realliyifei/ResearchQA); the out-of-domain evaluation uses [HealthBench](https://github.com/openai/simple-evals) and [ResearchQA](https://github.com/realliyifei/ResearchQA) with their own grader prompts; the recall and rubric-free judge prompts follow Mahmoud et al. (2026) and the second judge prompt is MT-Bench's (Zheng et al., 2023). The training grader is [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) and the base policy is [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

## ✉️ Contact
If you have any questions or feedback, feel free to reach out:
- Hyun Ryu: hyunr@andrew.cmu.edu
- Jong-Heon Kim: jongheon@kaist.ac.kr
