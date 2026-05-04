import re
from pathlib import Path

import pandas as pd
import yaml


def get_experiment_paths(results_dir="./results", filter_finished=False, filter_evaluated=False):
    results = Path(results_dir)
    seen, paths = set(), []
    for anchor in results.rglob("prompts_per_step.parquet"):
        exp = anchor.parent
        if str(exp) in seen:
            continue
        seen.add(str(exp))
        if filter_finished and not (exp / ".finished").exists():
            continue
        if filter_evaluated and not (exp / "eval" / "scores_per_step.parquet").exists():
            continue
        paths.append(str(exp.relative_to(results)))
    return sorted(paths)


def load_experiments_df(
    paths=None,
    results_dir="./results",
    source="eval",
    filter_finished=False,
    filter_evaluated=False,
):
    results_dir = Path(results_dir)

    if paths is None:
        paths = get_experiment_paths(
            results_dir=str(results_dir),
            filter_finished=filter_finished,
            filter_evaluated=filter_evaluated,
        )
    elif isinstance(paths, str):
        paths = [paths]

    parquet_name = (
        "prompts_per_step.parquet" if source == "train" else "eval/scores_per_step.parquet"
    )

    frames = []
    for rel_path in paths:
        exp_dir = results_dir / rel_path
        parquet_path = exp_dir / parquet_name

        if not parquet_path.exists():
            continue

        df = pd.read_parquet(parquet_path)
        if source != "train":
            df = df.rename(columns={"agent_prompt_set": "prompts", "test_score": "eval_score"})
        elif "prompt" in df.columns and "prompts" not in df.columns:
            df = df.rename(columns={"prompt": "prompts"})
        df["experiment_path"] = rel_path

        # always initialise columns so they exist even when the pattern doesn't match
        df["experiment.dataset"] = None
        df["experiment.optimizer"] = None
        df["experiment.seed"] = None

        m = re.match(r"dataset_(\w+)_optimizer_(\w+)_seed_(\d+)", exp_dir.name)
        if not m:
            m = re.match(r"seed_(\d+)_dataset_(\w+)_optimizer_(\w+)", exp_dir.name)
            if m:
                seed_v, ds_v, opt_v = m.groups()
                df["experiment.dataset"] = ds_v
                df["experiment.optimizer"] = opt_v
                df["experiment.seed"] = int(seed_v)
        else:
            ds_v, opt_v, seed_v = m.groups()
            df["experiment.dataset"] = ds_v
            df["experiment.optimizer"] = opt_v
            df["experiment.seed"] = int(seed_v)

        eval_scores_path = exp_dir / "eval" / "scores_per_step.parquet"
        if source == "train":
            if eval_scores_path.exists():
                eval_df = pd.read_parquet(eval_scores_path).rename(
                    columns={"agent_prompt_set": "prompts", "test_score": "eval_score"}
                )
                df = df.merge(eval_df[["prompts", "eval_score"]], on="prompts", how="left")
            else:
                df["eval_score"] = float("nan")

        eval_tokens_path = exp_dir / "eval" / "token_usage.yaml"
        if eval_tokens_path.exists():
            with open(eval_tokens_path) as f:
                token_usage = yaml.safe_load(f)
            df["eval_input_tokens"] = token_usage.get("input_tokens")
            df["eval_output_tokens"] = token_usage.get("output_tokens")
            df["eval_total_tokens"] = token_usage.get("total_tokens")
        else:
            df["eval_input_tokens"] = float("nan")
            df["eval_output_tokens"] = float("nan")
            df["eval_total_tokens"] = float("nan")

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    return df


def load_main_results_df(results_dir="./results_final/main_experiment") -> pd.DataFrame:
    """Load eval scores for all main-experiment runs into a tidy DataFrame.

    Columns: seed (int), dataset (str), optimizer (str, uppercased), score (float).
    The optimizer is read from config.yaml where available; "initial" runs are
    identified by directory name and never overridden by config.
    """
    rows = []
    for p in Path(results_dir).glob("*/eval/scores_per_step.parquet"):
        run_dir = p.parent.parent

        m = re.match(r"dataset_(\w+)_optimizer_(\w+)_seed_(\d+)", run_dir.name)
        if not m:
            continue
        dataset, optimizer_raw, seed = m.groups()

        optimizer = "INITIAL" if optimizer_raw == "initial" else optimizer_raw.upper()

        score = pd.read_parquet(p)["test_score"].iloc[0]

        token_path = run_dir / "eval" / "token_usage.yaml"
        if token_path.exists():
            with open(token_path) as f:
                tok = yaml.safe_load(f)
            input_tokens = tok.get("input_tokens", float("nan"))
            output_tokens = tok.get("output_tokens", float("nan"))
            total_tokens = tok.get("total_tokens", float("nan"))
        else:
            input_tokens = output_tokens = total_tokens = float("nan")

        prompts_path = run_dir / "prompts_per_step.parquet"
        if prompts_path.exists():
            pps = pd.read_parquet(prompts_path)
            valid = pps[pps["total_tokens"] <= 10_000_000]
            n_steps = int(valid["step"].max()) if not valid.empty else 0
        else:
            n_steps = None

        rows.append(
            {
                "seed": int(seed),
                "dataset": dataset,
                "optimizer": optimizer,
                "score": score,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "n_steps": n_steps,
            }
        )
    return pd.DataFrame(rows)


def load_trajectory_df(results_dir="results_final/main_experiment", dataset=None) -> pd.DataFrame:
    """Load per-step optimization trajectories for all runs in results_dir.

    Returns a tidy DataFrame with columns:
        step, total_tokens, score, optimizer, dataset, seed
    plus token-breakdown columns from prompts_per_step where available.

    Loading strategy per optimizer:
    - INITIAL: eval/scores_per_step.parquet only; step=0, total_tokens=0.
    - MIPRO/others without eval_all_final:
        prompts_per_step LEFT JOIN eval/scores_per_step.
    - CANTANTE/GEPA (with eval_all_final):
        prompts_per_step LEFT JOIN eval_all_final/scores_per_step,
      then eval/scores_per_step is merged on top with higher priority.
    """
    results_dir = Path(results_dir)
    frames = []

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        m = re.match(r"seed_(\d+)_dataset_(\w+)_optimizer_(.+)", run_dir.name)
        if m:
            seed, ds, optimizer_raw = m.groups()
        else:
            m = re.match(r"dataset_(\w+)_optimizer_(\w+)_seed_(\d+)", run_dir.name)
            if not m:
                continue
            ds, optimizer_raw, seed = m.groups()
        seed = int(seed)

        if dataset is not None and ds != dataset:
            continue

        optimizer = "INITIAL" if optimizer_raw == "initial" else optimizer_raw.upper()

        if optimizer == "INITIAL":
            eval_path = run_dir / "eval" / "scores_per_step.parquet"
            if not eval_path.exists():
                continue
            df = pd.read_parquet(eval_path).drop(columns=["agent_prompt_set"], errors="ignore")
            df = df.rename(columns={"test_score": "score"})
            df["step"] = 0
            df["total_tokens"] = 0
        else:
            prompts_path = run_dir / "prompts_per_step.parquet"
            if not prompts_path.exists():
                continue
            eval_all_path = run_dir / "eval_all" / "scores_per_step.parquet"
            eval_path = run_dir / "eval" / "scores_per_step.parquet"
            if not eval_all_path.exists() and not eval_path.exists():
                continue
            prompts_df = pd.read_parquet(prompts_path)
            if eval_all_path.exists():
                eval_all_df = pd.read_parquet(eval_all_path).rename(
                    columns={"agent_prompt_set": "prompt", "test_score": "score"}
                )
                df = prompts_df.merge(eval_all_df[["prompt", "score"]], on="prompt", how="left")
                if eval_path.exists():
                    eval_df = pd.read_parquet(eval_path).rename(
                        columns={"agent_prompt_set": "prompt", "test_score": "eval_score"}
                    )
                    df = df.merge(eval_df[["prompt", "eval_score"]], on="prompt", how="left")
                    df["score"] = df["eval_score"].combine_first(df["score"])
                    df = df.drop(columns=["eval_score"])
            else:
                eval_df = pd.read_parquet(eval_path).rename(
                    columns={"agent_prompt_set": "prompt", "test_score": "score"}
                )
                df = prompts_df.merge(eval_df[["prompt", "score"]], on="prompt", how="left")
            df["prompt_idx"] = range(len(df))
            df = df.drop(columns=["prompt"])

        df["optimizer"] = optimizer
        df["dataset"] = ds
        df["seed"] = seed
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_ranks(df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """Compute average optimizer ranks per (dataset) from a load_main_results_df DataFrame.

    Ranks are assigned per (dataset, seed) group (1 = best score), then averaged
    across seeds — a Friedman-style average rank.

    Returns a DataFrame with columns: optimizer, dataset, mean_rank.
    """
    df = df.copy()
    df["rank"] = df.groupby(["dataset", "seed"])["score"].rank(ascending=ascending, method="min")
    return df.groupby(["optimizer", "dataset"])["rank"].mean().reset_index(name="mean_rank")
