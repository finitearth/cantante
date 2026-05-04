"""
Generate all paper figures and save them to figures/.

"""

import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats

from src.analysis import load_main_results_df, load_trajectory_df
from src.analysis.style import set_style

set_style()
os.makedirs("figures", exist_ok=True)

COLORS = {
    "INITIAL": "#7A7A7A",
    "GEPA": "#8FBF8F",
    "MIPRO": "#7DB7C9",
    "CANTANTE": "#E76F51",
}
MARKERS = {"INITIAL": "o", "GEPA": "s", "MIPRO": "*", "CANTANTE": "D"}
MARKER_SIZES = {"INITIAL": 90, "GEPA": 105, "MIPRO": 230, "CANTANTE": 125}
ALPHAS = {"INITIAL": 0.65, "GEPA": 0.78, "MIPRO": 0.85, "CANTANTE": 0.98}
ZORDERS = {"INITIAL": 3, "GEPA": 4, "MIPRO": 4, "CANTANTE": 6}
OPTIMIZER_ORDER = ["INITIAL", "GEPA", "MIPRO", "CANTANTE"]

MARKER_SCALE = 1.3  # multiplies all MARKER_SIZES values
TRAJ_LINEWIDTH = 2.5  # line width in trajectory plot
TRAJ_MARKERSIZE = 14  # MIPRO star in trajectory plot (matplotlib markersize units)
LEGEND_MARKERSIZE = 14  # legend handles (non-MIPRO); MIPRO gets +2
FONT_ANNOT = 20  # "→ X.Xk / M" annotation labels
FONT_AXIS_LABEL = 20  # x/y axis labels in pareto plots
FONT_TITLE = 20  # per-dataset title in pareto plots
FONT_TICK = 20  # tick labels in pareto plots
FONT_LEGEND = 18  # legend text

FIG_TRAJ = (8, 6)  # trajectory plot (width, height) in inches
FIG_PARETO = (5, 4.6)  # pareto plots (width, height) in inches
FIG_LEGEND = (8, 0.6)  # standalone legend figure (width, height) in inches

# Trajectory plot
XLIM = 10_000_000
ALPHA = 0.1
DDOF = 1
CLAMP_X = XLIM * 0.87

df_traj = load_trajectory_df(results_dir="results/main_experiments", dataset="mbpp")

_last = (
    df_traj[df_traj["optimizer"].isin(["CANTANTE", "GEPA"]) & (df_traj["total_tokens"] < XLIM)]
    .sort_values("total_tokens")
    .groupby(["optimizer", "dataset", "seed"], as_index=False)
    .last()
    .assign(total_tokens=XLIM, step=9999, prompt_idx=9999)
)
df_traj = pd.concat([df_traj, _last], ignore_index=True)

plot_df = df_traj[(df_traj["total_tokens"] <= XLIM) | (df_traj["optimizer"] == "MIPRO")]

fig, ax = plt.subplots(figsize=FIG_TRAJ)

for optimizer, grp in plot_df.groupby("optimizer"):
    color = COLORS.get(optimizer)
    n_seeds = grp["seed"].nunique()

    if optimizer == "INITIAL":
        mean_score = grp["score"].mean()
        std_score = grp["score"].std(ddof=DDOF)
        ax.axhline(
            y=mean_score, color=color, linestyle="--", linewidth=TRAJ_LINEWIDTH, label=optimizer
        )
        ax.fill_between(
            [0, XLIM], mean_score - std_score, mean_score + std_score, color=color, alpha=ALPHA
        )

    elif optimizer == "MIPRO":
        scored = grp[grp["score"].notna()]
        for i, (_, row) in enumerate(scored.iterrows()):
            x = CLAMP_X if row["total_tokens"] >= XLIM else row["total_tokens"]
            ax.plot(
                x,
                row["score"],
                marker="*",
                linestyle="none",
                markersize=TRAJ_MARKERSIZE,
                color=color,
                label=optimizer if i == 0 else "_nolegend_",
                zorder=5,
            )
            if row["total_tokens"] >= XLIM:
                ax.annotate(
                    f"→ {row['total_tokens'] / 1e6:.1f}M",
                    xy=(x, row["score"]),
                    xytext=(9, -1),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=FONT_ANNOT,
                    color=color,
                )

    else:  # CANTANTE, GEPA
        pivot = (
            grp.pivot_table(index="total_tokens", columns="seed", values="score", aggfunc="first")
            .sort_index()
            .ffill()
            .dropna()
        )
        agg = pd.DataFrame(
            {
                "total_tokens": pivot.index,
                "mean_score": pivot.mean(axis=1).values,
                "std_score": pivot.std(axis=1, ddof=DDOF).values,
            }
        )
        ax.plot(
            agg["total_tokens"],
            agg["mean_score"],
            drawstyle="steps-post",
            linestyle="-",
            linewidth=TRAJ_LINEWIDTH,
            label=optimizer,
            color=color,
        )
        ax.fill_between(
            agg["total_tokens"],
            agg["mean_score"] - agg["std_score"],
            agg["mean_score"] + agg["std_score"],
            alpha=ALPHA,
            color=color,
            edgecolor="w",
            step="post",
        )

ax.set_xlim(0, XLIM)
ax.set_xticks([i * 1_000_000 for i in range(1, 11)])
ax.set_xticklabels([str(i) for i in range(1, 11)])
ax.set_xlabel("M Tokens", fontsize=FONT_AXIS_LABEL)
ax.set_ylabel("Score", fontsize=FONT_AXIS_LABEL)
ax.tick_params(axis="both", labelsize=FONT_TICK)
ax.legend(fontsize=FONT_LEGEND, loc="upper left")
ax.set_ylim(bottom=None, top=ax.get_ylim()[1] * 1.25)
plt.tight_layout()
plt.savefig("figures/trajectory_mbpp.pdf")
plt.close()

# Pareto plots
N_EVAL = 500
DATASET_ORDER = ["gsm8k", "hotpotqa", "mbpp"]
X_LIMITS = {
    "mbpp": (1.5, 2.6),
    "gsm8k": (1.5, 2.6),
    "hotpotqa": (1, 2.6),
}
X_TICKS = [1.5, 2.0, 2.5]

df_main = load_main_results_df(results_dir="results/main_experiments")
df_main["ktokens_per_query"] = df_main["total_tokens"] / N_EVAL / 1000

for dataset in DATASET_ORDER:
    x_min, x_max = X_LIMITS[dataset]
    clamp_x = x_min + (x_max - x_min) * 0.87

    df_ds = df_main[df_main["dataset"] == dataset]

    fig, ax = plt.subplots(figsize=FIG_PARETO)

    for optimizer in OPTIMIZER_ORDER:
        grp = df_ds[df_ds["optimizer"] == optimizer]
        if grp.empty:
            continue

        for _, row in grp.iterrows():
            x = row["ktokens_per_query"]
            y = row["score"]
            clamped = x >= x_max
            plot_x = clamp_x if clamped else x

            edgecolor = "white"
            linewidth = 0.6
            if optimizer == "MIPRO":
                edgecolor = COLORS[optimizer]
                linewidth = 0.2

            ax.scatter(
                plot_x,
                y,
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                s=MARKER_SIZES[optimizer] * MARKER_SCALE,
                alpha=ALPHAS[optimizer],
                zorder=ZORDERS[optimizer],
                edgecolors=edgecolor,
                linewidths=linewidth,
            )

            if clamped:
                ax.annotate(
                    f"→ {x:.1f}k",
                    xy=(plot_x, y),
                    xytext=(7, -1),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=FONT_ANNOT,
                    color=COLORS[optimizer],
                    alpha=ALPHAS[optimizer],
                )

    if dataset == "mbpp":
        ax.annotate(
            "better",
            xy=(0.07, 0.93),
            xytext=(0.25, 0.75),
            xycoords="axes fraction",
            textcoords="axes fraction",
            ha="center",
            fontsize=FONT_ANNOT,
            color="darkorchid",
            arrowprops=dict(arrowstyle="fancy", color="darkorchid", lw=2.0),
        )

    ax.set_xlim(x_min, x_max)
    ax.set_xticks(X_TICKS)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.set_xlabel("kTokens per query", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Score", fontsize=FONT_AXIS_LABEL)
    # ax.set_title(dataset.upper(), fontsize=FONT_TITLE, pad=8)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.minorticks_on()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"figures/pareto_{dataset}.pdf")
    plt.close()

# Legend
legend_handles = [
    Line2D(
        [0],
        [0],
        marker=MARKERS[optimizer],
        linestyle="None",
        label=optimizer,
        markerfacecolor=COLORS[optimizer],
        markeredgecolor="white",
        markeredgewidth=0.6,
        markersize=LEGEND_MARKERSIZE if optimizer != "MIPRO" else LEGEND_MARKERSIZE + 2,
        alpha=ALPHAS[optimizer],
    )
    for optimizer in OPTIMIZER_ORDER
]

fig_legend, ax_legend = plt.subplots(figsize=FIG_LEGEND)
ax_legend.axis("off")
ax_legend.legend(
    handles=legend_handles,
    loc="center",
    ncol=4,
    frameon=False,
    fontsize=FONT_LEGEND,
    handletextpad=0.6,
    columnspacing=1.8,
)
plt.savefig("figures/pareto_legend.pdf", bbox_inches="tight")
plt.close()

# Credit correlation analysis
AGENT_ORDER_HEATMAP = {
    "gsm8k": ["executor_1", "executor_2", "executor_3", "consensus"],
    "hotpotqa": ["retriever", "reader", "hallucination-det.", "synthesizer"],
    "mbpp": ["planner", "coder", "validator"],
}


def _normalize_instruction(text):
    text = re.sub(r"<few_shots_section>.*?</few_shots_section>", "", text, flags=re.DOTALL)
    return text.replace("{few_shots}", "")


run_dirs = sorted(
    d
    for d in Path("results/main_experiments").iterdir()
    if re.search(r"cantante", d.name, re.IGNORECASE)
)

rows = []
for run_dir in run_dirs:
    seed = int(re.search(r"seed_(\d+)", run_dir.name).group(1))
    dataset = re.search(r"dataset_(\w+)_optimizer", run_dir.name).group(1)
    for ckpt_path in sorted((run_dir / "checkpoints").glob("checkpoint_step_*.json")):
        step = int(re.search(r"(\d+)", ckpt_path.stem).group(1))
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        credits = {
            agent: {
                _normalize_instruction(p["instruction"]): s
                for p, s in zip(data["prompts"], data["scores"])
            }
            for agent, data in ckpt.get("agent_optimizers", {}).items()
        }
        for prompt_str, score in zip(ckpt["prompts"], ckpt["scores"]):
            prompt = json.loads(prompt_str)
            row = {"seed": seed, "dataset": dataset, "step": step, "train_score": score}
            for agent, instruction in prompt.items():
                row[agent] = instruction
                row[f"{agent}_credit"] = credits.get(agent, {}).get(instruction)
            rows.append(row)

df_raw = pd.DataFrame(rows)
credit_cols = [c for c in df_raw.columns if c.endswith("_credit")]
frames = []
for dataset, df_ds in df_raw.groupby("dataset"):
    cols = [c for c in credit_cols if df_ds[c].notna().any()]
    frames.append(df_ds.dropna(subset=cols))
df_ckpt = pd.concat(frames).reset_index(drop=True)

corr_rows = []
for dataset in sorted(df_ckpt["dataset"].unique()):
    df_ds = df_ckpt[df_ckpt["dataset"] == dataset]
    for seed in sorted(df_ds["seed"].unique()):
        df_seed = df_ds[df_ds["seed"] == seed]
        for agent in [c.replace("_credit", "") for c in credit_cols]:
            col = f"{agent}_credit"
            valid = df_seed[[col, "train_score"]].dropna()
            if valid.empty:
                continue
            rho, _ = stats.spearmanr(valid[col], valid["train_score"])
            corr_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "agent": agent,
                    "spearman_rho": round(rho, 2),
                }
            )

df_corr = pd.DataFrame(corr_rows)

for dataset in sorted(df_corr["dataset"].unique()):
    pivot = df_corr[df_corr["dataset"] == dataset].pivot(
        index="seed", columns="agent", values="spearman_rho"
    )
    pivot.columns = [
        c.replace("_agent", "")
        .replace("hallucination_detector", "hallucination-det.")
        .replace("_", " ")
        .title()
        for c in pivot.columns
    ]
    _order = [
        a.replace("_", " ").title()
        for a in AGENT_ORDER_HEATMAP.get(dataset, [])
        if a.replace("_", " ").title() in pivot.columns
    ] or list(pivot.columns)
    pivot = pivot[_order]

    fig, ax = plt.subplots(figsize=(max(5, len(pivot.columns) * 1.4), len(pivot.index) * 1.2 + 1))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="coolwarm",
        vmin=-0.6,
        vmax=0.6,
        annot=True,
        fmt=".2f",
        annot_kws={"color": "black", "fontsize": 18},
        linewidths=0.5,
        cbar=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    x_rot = 45 if dataset in ("gsm8k", "hotpotqa") else 0
    ax.set_xticklabels(ax.get_xticklabels(), rotation=x_rot, ha="right" if x_rot else "center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(which="both", length=0, labelsize=18)
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=0)
    plt.tight_layout()
    plt.savefig(f"figures/credit_corr_{dataset}.pdf", bbox_inches="tight")
    plt.close()
