"""The run configuration behind scripts/train.sh.

A run is one yaml under configs/train/ with the reward's switches:

    run_id:    the run's name (its directory under outputs/train/)
    reward:    recall | fbeta          recall = the standard rubric-coverage reward
    beta:      a positive number       fbeta only: F_beta's beta (1.0, 1.25, ...)
    precision: credit | generic        fbeta only: FORK's credit-based precision, or
                                       the Generic Criteria labeler
    domain:    medical | science       the prepared split under data/<domain>/ and
                                       the prompts' domain wording
    epochs:    optional                 overrides configs/grpo.yaml

The GRPO hyperparameters shared by every run live in configs/grpo.yaml. This
module validates a run, exports its switches to the launcher's environment
(`python -m fork.launch env <run.yaml>`) and renders verl's command line
(`python -m fork.launch args <run.yaml> ...`), so the launcher stays a thin
shell script and the arguments can be checked without a GPU.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import yaml

__all__ = ["REWARDS", "PRECISIONS", "DOMAINS", "load_grpo", "load_run", "verl_args"]

REWARDS = ("recall", "fbeta")
PRECISIONS = ("credit", "generic")
DOMAINS = ("medical", "science")
ROOT = Path(__file__).resolve().parents[2]
GRPO_YAML = ROOT / "configs" / "grpo.yaml"


def load_grpo(path: str | Path = GRPO_YAML) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    need = ("policy", "rollouts_per_prompt", "prompts_per_step", "temperature", "max_prompt_tokens",
            "max_response_tokens", "lr", "lr_warmup_ratio", "weight_decay", "kl_coef", "kl_loss_type",
            "epochs", "checkpoint_every")
    missing = [k for k in need if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    return cfg


def load_run(path: str | Path) -> dict:
    """The validated run config. Every switch value is checked here, at launch,
    so a misconfiguration dies before the model loads; the reward function
    checks the same values again from its environment at the first batch."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(cfg) - {"run_id", "reward", "beta", "precision", "domain", "epochs"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    run_id = cfg.get("run_id")
    if not isinstance(run_id, str) or not run_id or "/" in run_id:
        raise ValueError(f"{path}: run_id must be a non-empty name, got {run_id!r}")
    reward = cfg.get("reward")
    if reward not in REWARDS:
        raise ValueError(f"{path}: reward must be {'|'.join(REWARDS)}, got {reward!r}")
    beta, precision = cfg.get("beta"), cfg.get("precision")
    if reward == "fbeta":
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not beta > 0:
            raise ValueError(f"{path}: reward: fbeta needs beta (a number > 0), got {beta!r}")
        if precision not in PRECISIONS:
            raise ValueError(f"{path}: reward: fbeta needs precision {'|'.join(PRECISIONS)}, got {precision!r}")
    else:
        if beta is not None or precision is not None:
            raise ValueError(f"{path}: beta and precision apply to reward: fbeta only")
    domain = cfg.get("domain", "medical")
    if domain not in DOMAINS:
        raise ValueError(f"{path}: domain must be {'|'.join(DOMAINS)}, got {domain!r}")
    if precision == "generic" and domain != "medical":
        raise ValueError(f"{path}: the Generic Criteria prompt is medical-only; precision: generic "
                         f"needs domain: medical")
    epochs = cfg.get("epochs")
    if epochs is not None and (isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0):
        raise ValueError(f"{path}: epochs must be a positive integer, got {epochs!r}")
    return {"run_id": run_id, "reward": reward, "beta": float(beta) if beta is not None else None,
            "precision": precision, "domain": domain, "epochs": epochs}


def verl_args(run: dict, grpo: dict, *, train_files: str, val_files: str, n_gpus: int,
              micro_bsz: int, logp_micro_bsz: int, rollout_gpu_mem: float, optim_offload: bool,
              logger: str, project: str, ckpt_dir: str, run_dir: str, reward_path: str,
              total_steps: int | None = None) -> list[str]:
    """verl's command-line overrides for one run: the GRPO hyperparameters of
    configs/grpo.yaml, the node geometry, the reward wiring. The trainer's
    global batch — prompts_per_step × rollouts_per_prompt sequences per step —
    must divide across the trainer GPUs (verl asserts it)."""
    seqs = grpo["prompts_per_step"] * grpo["rollouts_per_prompt"]
    if seqs % n_gpus:
        raise ValueError(f"{seqs} rollouts per step do not divide across {n_gpus} trainer GPUs")
    epochs = run.get("epochs") or grpo["epochs"]
    args = [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files={train_files}",
        f"data.val_files={val_files}",
        f"data.train_batch_size={grpo['prompts_per_step']}",
        f"data.max_prompt_length={grpo['max_prompt_tokens']}",
        f"data.max_response_length={grpo['max_response_tokens']}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={grpo['policy']}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"actor_rollout_ref.actor.optim.lr={grpo['lr']}",
        "actor_rollout_ref.actor.optim.lr_scheduler_type=constant",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps_ratio={grpo['lr_warmup_ratio']}",
        f"actor_rollout_ref.actor.optim.weight_decay={grpo['weight_decay']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={grpo['prompts_per_step']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro_bsz}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={grpo['kl_coef']}",
        f"actor_rollout_ref.actor.kl_loss_type={grpo['kl_loss_type']}",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={optim_offload}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={grpo['rollouts_per_prompt']}",
        f"actor_rollout_ref.rollout.temperature={grpo['temperature']}",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={rollout_gpu_mem}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={logp_micro_bsz}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={logp_micro_bsz}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        f"custom_reward_function.path={reward_path}",
        "custom_reward_function.name=compute_score",
        "reward_model.reward_manager=batch",
        "reward_model.use_reward_loop=False",
        "trainer.critic_warmup=0",
        f"trainer.logger={logger}",
        f"trainer.project_name={project}",
        f"trainer.experiment_name={run['run_id']}",
        f"trainer.n_gpus_per_node={n_gpus}",
        "trainer.nnodes=1",
        f"trainer.save_freq={grpo['checkpoint_every']}",
        "trainer.test_freq=-1",
        "trainer.val_before_train=False",
        f"trainer.total_epochs={epochs}",
        f"trainer.default_local_dir={ckpt_dir}",
        f"hydra.run.dir={run_dir}/hydra/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}",
    ]
    if total_steps is not None:
        args.append(f"trainer.total_training_steps={total_steps}")
    return args


def _env_lines(run: dict, grpo: dict) -> list[str]:
    """`export K=V` lines for scripts/train.sh: the run's switches and the data
    paths the domain implies (data/<domain>/verl/{train,eval}.parquet)."""
    kv = {
        "RUN_ID": run["run_id"],
        "FORK_REWARD": run["reward"],
        "FORK_BETA": "" if run["beta"] is None else repr(run["beta"]),
        "FORK_PRECISION": run["precision"] or "",
        "FORK_DOMAIN": run["domain"],
        "EPOCHS": str(run.get("epochs") or grpo["epochs"]),
        "TRAIN_FILES": f"data/{run['domain']}/verl/train.parquet",
        "VAL_FILES": f"data/{run['domain']}/verl/eval.parquet",
        "POLICY": grpo["policy"],
    }
    return [f"export {k}={shlex.quote(v)}" for k, v in kv.items()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("env", help="print `export` lines for the run's switches")
    e.add_argument("run")
    a = sub.add_parser("args", help="print verl's command-line overrides, one per line")
    a.add_argument("run")
    a.add_argument("--n-gpus", type=int, required=True)
    a.add_argument("--micro-bsz", type=int, required=True)
    a.add_argument("--logp-micro-bsz", type=int, required=True)
    a.add_argument("--rollout-gpu-mem", type=float, required=True)
    a.add_argument("--optim-offload", choices=("True", "False"), required=True)
    a.add_argument("--logger", required=True)
    a.add_argument("--project", required=True)
    a.add_argument("--ckpt-dir", required=True)
    a.add_argument("--run-dir", required=True)
    a.add_argument("--reward-path", required=True)
    a.add_argument("--total-steps", type=int, default=None)
    a.add_argument("--grpo", default=str(GRPO_YAML))
    ns = ap.parse_args(argv)
    run = load_run(ns.run)
    if ns.cmd == "env":
        print("\n".join(_env_lines(run, load_grpo())))
        return
    grpo = load_grpo(ns.grpo)
    for line in verl_args(run, grpo, train_files=f"data/{run['domain']}/verl/train.parquet",
                          val_files=f"data/{run['domain']}/verl/eval.parquet", n_gpus=ns.n_gpus,
                          micro_bsz=ns.micro_bsz, logp_micro_bsz=ns.logp_micro_bsz,
                          rollout_gpu_mem=ns.rollout_gpu_mem, optim_offload=ns.optim_offload == "True",
                          logger=ns.logger, project=ns.project, ckpt_dir=ns.ckpt_dir, run_dir=ns.run_dir,
                          reward_path=ns.reward_path, total_steps=ns.total_steps):
        print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
