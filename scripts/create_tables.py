"""Generate LaTeX tables for the paper.

Output files are written to tables/. All scores are derived from experiment
results in results/.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import (
    LatexTable,
    compute_ranks,
    load_experiments_df,
    load_main_results_df,
    render_table,
)

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Paper-specific ordering and labels ──────────────────────────────────────
OPTIMIZER_ORDER = ["INITIAL", "GEPA", "MIPRO", "CANTANTE"]
OPTIMIZER_LABELS = {
    "INITIAL": "Initial",
    "GEPA": "GEPA",
    "MIPRO": "MIPROv2",
    "CANTANTE": "\\alg{} (ours)",
}
DATASET_ORDER = ["mbpp", "gsm8k", "hotpotqa"]
DATASET_LABELS = {"gsm8k": "GSM8K", "hotpotqa": "HotpotQA", "mbpp": "MBPP"}

# ── Ablation configs ─────────────────────────────────────────────────────────
MAIN_RESULTS_DIR = "./results/main_experiments"
ABLATIONS_RESULTS_DIR = "./results/ablations"

ABLATION_GROUPS = [
    {
        "label": "Identity Attribution",
        "entries": [
            {"path": "identity_attribution", "label": "Equal Steps", "eval_dir": "eval_step_5"},
            {"path": "identity_attribution", "label": "Equal Budget", "eval_dir": "eval"},
        ],
    },
    {
        "label": "Attribution component",
        "entries": [
            {"path": "alternative_attribution_prompt", "label": "alt. Prompt", "eval_dir": "eval"},
            {"path": "alternative_attribution_model", "label": "alt. Model", "eval_dir": "eval"},
        ],
    },
    {
        "label": "Prompt optimizer",
        "entries": [
            {"path": "enabling_fs", "label": "w/ Few-Shots", "eval_dir": "eval"},
            {"path": "alternative_local_optim", "label": "alt. Optimizer", "eval_dir": "eval"},
        ],
    },
    {
        "label": "Group size",
        "entries": [
            {"path": "group_size/group_size_2_7c8ea1", "label": "g = 2", "eval_dir": "eval"},
            {"path": "group_size/group_size_5_b6f8e1", "label": "g = 5", "eval_dir": "eval"},
        ],
    },
    {
        "label": "Dataset size",
        "entries": [
            {
                "path": "ds_size/dev_size_75_c08d49",
                "label": "|DS| = \\phantom{0}75",
                "eval_dir": "eval",
            },
            {"path": "ds_size/dev_size_150_0c2496", "label": "|DS| = 150", "eval_dir": "eval"},
        ],
    },
]

ABLATION_LABELS = [entry for group in ABLATION_GROUPS for entry in group["entries"]]

HIGHLIGHT_TABLES = {
    "tables/main_results_table.tex",
    "tables/main_token_table.tex",
}


# ── Cell formatting ──────────────────────────────────────────────────────────
def _fmt_cell(mean: float, std: float, scale: float = 100) -> str:
    if np.isnan(mean):
        return "—"
    m = mean * scale
    s = std * scale
    m_str = ("\\phantom{0}" if m < 10 else "") + f"{m:.2f}"
    if np.isnan(s):
        return m_str
    s_str = ("\\phantom{0}" if s < 10 else "") + f"{s:.2f}"
    return f"{m_str}\\,$_{{\\pm{s_str}}}$"


# ── Table builders ───────────────────────────────────────────────────────────
def make_main_table(df: pd.DataFrame) -> LatexTable:
    agg = df.groupby(["optimizer", "dataset"])["score"].agg(["mean", lambda s: s.std(ddof=1)])
    agg.columns = ["mean", "std"]

    dataset_cols = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    headers = ["Optimizer"] + [DATASET_LABELS[d] for d in dataset_cols] + ["Avg Rank"]
    avg_rank = compute_ranks(df).groupby("optimizer")["mean_rank"].mean()

    rows, values = [], []
    for opt in OPTIMIZER_ORDER:
        if opt not in df["optimizer"].unique():
            continue
        cells = [OPTIMIZER_LABELS.get(opt, opt)]
        row_vals = []
        for d in dataset_cols:
            try:
                mean, std = float(agg.loc[(opt, d), "mean"]), float(agg.loc[(opt, d), "std"])
            except KeyError:
                mean, std = float("nan"), float("nan")
            cells.append(_fmt_cell(mean, std))
            row_vals.append(mean if not np.isnan(mean) else None)
        r_raw = avg_rank.get(opt)
        r = float(r_raw) if r_raw is not None else None
        cells.append(f"{r:.2f}" if r is not None else "—")
        row_vals.append(-r if r is not None else None)
        rows.append(tuple(cells))
        values.append(row_vals)

    baselines = [o for o in OPTIMIZER_ORDER if o != "CANTANTE" and o in df["optimizer"].unique()]
    midrule_after_idx = len(baselines) - 1

    def _colored(val: float, higher_is_better: bool = True) -> str:
        color = "cbGreen" if (val > 0) == higher_is_better else "cbOrange"
        return f"\\textcolor{{{color}}}{{{val:.2f}}}"

    def _delta_row(label: str, reference: str) -> tuple:
        cells = [label]
        cantante_rank = avg_rank.get("CANTANTE")
        ref_rank = avg_rank.get(reference)
        for d in dataset_cols:
            c_mean = (
                float(agg.loc[("CANTANTE", d), "mean"])
                if ("CANTANTE", d) in agg.index
                else float("nan")
            )
            r_mean = (
                float(agg.loc[(reference, d), "mean"])
                if (reference, d) in agg.index
                else float("nan")
            )
            cells.append(
                "—" if np.isnan(c_mean) or np.isnan(r_mean) else _colored((c_mean - r_mean) * 100)
            )
        if cantante_rank is not None and ref_rank is not None:
            cells.append(_colored(float(cantante_rank) - float(ref_rank), higher_is_better=False))
        else:
            cells.append("—")
        return tuple(cells)

    rows.append(_delta_row("$\\Delta$ vs. Initial", "INITIAL"))
    values.append([None] * (len(dataset_cols) + 1))

    best_baseline_rank = min(
        (float(avg_rank[o]) for o in baselines if o in avg_rank), default=float("nan")
    )
    best_cells = ["$\\Delta$ vs. Best"]
    for d in dataset_cols:
        c_mean = (
            float(agg.loc[("CANTANTE", d), "mean"])
            if ("CANTANTE", d) in agg.index
            else float("nan")
        )
        best = max(
            (float(agg.loc[(o, d), "mean"]) for o in baselines if (o, d) in agg.index),
            default=float("nan"),
        )
        best_cells.append(
            "—" if np.isnan(c_mean) or np.isnan(best) else _colored((c_mean - best) * 100)
        )
    cantante_rank = avg_rank.get("CANTANTE")
    best_cells.append(
        _colored(float(cantante_rank) - best_baseline_rank, higher_is_better=False)
        if cantante_rank is not None and not np.isnan(best_baseline_rank)
        else "—"
    )
    rows.append(tuple(best_cells))
    values.append([None] * (len(dataset_cols) + 1))

    return LatexTable(
        headers=headers,
        rows=rows,
        caption=(
            "Test accuracy (\\%) averaged over three seeds ($\\pm$std). "
            "\\textbf{Bold: best}, \\underline{underlined: second-best} per benchmark. "
            "Average rank computed across benchmarks and seeds. "
            "Bottom rows show \\alg's absolute gain over the initial prompt and best baseline."
        ),
        label="tab:main-results",
        midrule_after=[midrule_after_idx],
        values=values,
        higher_is_better=True,
        resizebox_width="0.85\\textwidth",
        col_format="l" + "l" * len(dataset_cols) + "c",
        array_stretch=1,
    )


def make_main_token_table(df: pd.DataFrame, n_eval: int = 500) -> LatexTable:
    dataset_cols = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    headers = ["Optimizer"] + [DATASET_LABELS[d] for d in dataset_cols] + ["Avg Rank"]

    per_inv = df.copy()
    for col in ["input_tokens", "output_tokens", "total_tokens"]:
        per_inv[col] = per_inv[col] / n_eval / 1000

    agg = (
        per_inv.groupby(["optimizer", "dataset"])["total_tokens"]
        .agg(mean="mean", std=lambda s: s.std(ddof=1))
        .reset_index()
    )
    token_rank_df = per_inv[["seed", "dataset", "optimizer", "total_tokens"]].rename(
        columns={"total_tokens": "score"}
    )
    avg_rank = compute_ranks(token_rank_df, ascending=True).groupby("optimizer")["mean_rank"].mean()

    rows, values = [], []
    for opt in OPTIMIZER_ORDER:
        if opt not in df["optimizer"].unique():
            continue
        cells = [OPTIMIZER_LABELS.get(opt, opt)]
        row_vals = []
        for d in dataset_cols:
            mask = (agg["optimizer"] == opt) & (agg["dataset"] == d)
            row = agg[mask]
            if row.empty or np.isnan(row["mean"].values[0]):
                cells.append("—")
                row_vals.append(None)
            else:
                mean, std = float(row["mean"].values[0]), float(row["std"].values[0])
                cells.append(_fmt_cell(mean, std, 1))
                row_vals.append(mean)
        r_raw = avg_rank.get(opt)
        r = float(r_raw) if r_raw is not None else None
        cells.append(f"{r:.2f}" if r is not None else "—")
        row_vals.append(r if r is not None else None)
        rows.append(tuple(cells))
        values.append(row_vals)

    return LatexTable(
        headers=headers,
        rows=rows,
        caption=(
            "Inference-time input-token usage on the evaluation sets. We report the "
            "mean number of input tokens per inference, in thousands, with standard "
            "deviation across three seeds (Bessel's correction). "
            "{\\textbf{Bold}} and {\\underline{underlined}} values indicate the "
            "lowest and second-lowest mean token usage for each benchmark, respectively."
        ),
        label="tab:token-usage",
        midrule_after=[len(rows) - 2],
        values=values,
        higher_is_better=False,
        resizebox_width="0.85\\textwidth",
        array_stretch=1.1,
    )


def make_frac_meta_token_table(df: pd.DataFrame) -> LatexTable:
    def _select_step(group):
        valid = group[group["total_tokens"] <= 10_000_000]
        subset = valid if not valid.empty else group
        return subset.sort_values("step").iloc[[-1]]

    filtered = (
        df.groupby("experiment_path", group_keys=False).apply(_select_step).reset_index(drop=True)
    )
    filtered["frac_meta_tokens"] = (
        (filtered["meta_input_tokens"] + filtered["meta_output_tokens"])
        / filtered["total_tokens"]
        * 100
    )

    dataset_cols = [d for d in DATASET_ORDER if d in filtered["experiment.dataset"].unique()]
    headers = ["Optimizer"] + [DATASET_LABELS[d] for d in dataset_cols]
    agg = (
        filtered.groupby(["experiment.optimizer", "experiment.dataset"])["frac_meta_tokens"]
        .agg(mean="mean", std=lambda s: s.std(ddof=1))
        .reset_index()
    )

    rows, values = [], []
    for opt in OPTIMIZER_ORDER:
        opt_lower = opt.lower()
        if opt_lower not in filtered["experiment.optimizer"].unique():
            continue
        cells = [OPTIMIZER_LABELS.get(opt, opt)]
        row_vals = []
        for d in dataset_cols:
            mask = (agg["experiment.optimizer"] == opt_lower) & (agg["experiment.dataset"] == d)
            row = agg[mask]
            if row.empty or np.isnan(row["mean"].values[0]):
                cells.append("—")
                row_vals.append(None)
            else:
                mean, std = float(row["mean"].values[0]), float(row["std"].values[0])
                cells.append(_fmt_cell(mean, std, scale=1))
                row_vals.append(mean)
        rows.append(tuple(cells))
        values.append(row_vals)

    return LatexTable(
        headers=headers,
        rows=rows,
        caption=(
            "Fraction of optimization tokens spent on meta (attribution/proposal) calls, "
            "as a percentage of total tokens used up to the token budget. "
            "Values are averaged across seeds."
        ),
        label="tab:frac-meta-tokens",
        col_format="l" + "c" * len(dataset_cols),
        values=values,
        higher_is_better=False,
    )


def make_ablation_table(
    groups: list[dict],
    results_dir: str,
    main_score: float | None = None,
    show_delta: bool = False,
) -> LatexTable:
    def fmt_abs(val: float | None) -> str:
        return f"{val * 100:.2f}" if val is not None else "—"

    def fmt_delta(val: float | None) -> str:
        if val is None or main_score is None:
            return "—"
        d = (val - main_score) * 100
        sign = "$+$" if d >= 0 else "$-$"
        pad = "\\phantom{0}" if abs(d) < 10 else ""
        return f"{sign}\\,{pad}{abs(d):.2f}"

    fmt_entry = fmt_delta if show_delta else fmt_abs

    def _load_score(path: str, eval_dir: str) -> float | None:
        p = Path(results_dir) / path / eval_dir / "scores_per_step.parquet"
        return float(pd.read_parquet(p)["test_score"].iloc[0]) if p.exists() else None

    rows = [("Default setting", fmt_abs(main_score) if main_score is not None else "xx.xx")]
    values = [[main_score]]
    midrule_after = [0]

    for g_idx, group in enumerate(groups):
        rows.append((f"\\textit{{{group['label']}}}", ""))
        values.append([None])
        for entry in group["entries"]:
            score = _load_score(entry["path"], entry["eval_dir"])
            rows.append((f"\\hspace{{1em}}{entry['label']}", fmt_entry(score)))
            values.append([score])
        if g_idx < len(groups) - 1:
            midrule_after.append(len(rows) - 1)

    score_col_header = "$\\Delta$ Score" if show_delta else "Score (\\%)"
    return LatexTable(
        headers=["Configuration", score_col_header],
        rows=rows,
        caption=(
            "Ablation results on GSM8K (Seed 42). "
            "Test accuracy (\\%) for configurations selected by development-set performance."
        ),
        label="tab:ablations",
        midrule_after=midrule_after,
        values=values,
        higher_is_better=True,
        col_format="lr",
        wrap_width="0.45\\textwidth",
    )


def make_ablation_token_table(
    df: pd.DataFrame,
    labels: list[dict],
    main_tokens: float | None = None,
    n_eval: int = 500,
) -> LatexTable:
    tokens = (
        df[df["eval_total_tokens"].notna()]
        .groupby("experiment_path")["eval_total_tokens"]
        .first()
        .to_dict()
    )

    def fmt(val: float | None) -> str:
        return "—" if val is None else _fmt_cell(val / n_eval / 1000, float("nan"), scale=1)

    rows = [("Default setting", fmt(main_tokens))]
    values = [[main_tokens / n_eval / 1000 if main_tokens is not None else None]]
    for entry in labels:
        tok = tokens.get(entry["path"])
        rows.append((entry["label"], fmt(tok) if tok is not None else "—"))
        values.append([tok / n_eval / 1000 if tok is not None else None])

    return LatexTable(
        headers=["Configuration", "Tokens/inv (k)"],
        rows=rows,
        caption=(
            "Evaluation token usage per MAS invocation for ablation configurations "
            "on GSM8K, seed 42. Values are total evaluation tokens divided by the "
            "500-sample evaluation set size, in thousands."
        ),
        label="tab:ablation-tokens",
        midrule_after=[0],
        values=values,
        higher_is_better=False,
    )


# ── Ablation reference configuration ─────────────────────────────────────────
# All ablations use this fixed setup; used to look up the main-experiment
# baseline score and token count for the delta column.
ABLATION_REF = {"dataset": "gsm8k", "optimizer": "CANTANTE", "seed": 42}


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_ablations = load_experiments_df(
        results_dir=ABLATIONS_RESULTS_DIR, filter_evaluated=True, source="eval"
    )
    df_main = load_main_results_df(results_dir=MAIN_RESULTS_DIR)
    df_train = load_experiments_df(
        results_dir=MAIN_RESULTS_DIR,
        filter_finished=False,
        filter_evaluated=True,
        source="train",
    )

    mask = (
        (df_main["optimizer"] == ABLATION_REF["optimizer"])
        & (df_main["dataset"] == ABLATION_REF["dataset"])
        & (df_main["seed"] == ABLATION_REF["seed"])
    )
    ref_row = df_main.loc[mask]
    main_score = float(ref_row["score"].iloc[0]) if not ref_row.empty else None
    main_tokens = float(ref_row["total_tokens"].iloc[0]) if not ref_row.empty else None

    tables = {
        "tables/main_results_table.tex": make_main_table(df_main),
        "tables/main_token_table.tex": make_main_token_table(df_main),
        "tables/frac_meta_token_table.tex": make_frac_meta_token_table(df_train),
        "tables/ablation_score_table.tex": make_ablation_table(
            ABLATION_GROUPS, ABLATIONS_RESULTS_DIR, main_score=main_score, show_delta=True
        ),
        "tables/ablation_token_table.tex": make_ablation_token_table(
            df_ablations, ABLATION_LABELS, main_tokens=main_tokens
        ),
    }

    for out_path, table in tables.items():
        path = Path(out_path)
        path.parent.mkdir(exist_ok=True)
        highlight = out_path in HIGHLIGHT_TABLES
        path.write_text(render_table(table, bold_best=highlight, underline_second=highlight) + "\n")
        print(f"Wrote {path}")
