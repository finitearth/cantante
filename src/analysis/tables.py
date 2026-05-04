from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


@dataclass
class LatexTable:
    headers: list[str]
    rows: list[tuple]
    caption: str
    label: str
    col_format: str | None = None
    midrule_after: list[int] | None = None
    # Numeric values parallel to rows, skipping the first (label) column.
    # Used by render_table to identify best/second-best per column.
    values: list[list[float | None]] | None = None
    higher_is_better: bool = True
    # Set to e.g. "0.75\\textwidth" to wrap the tabular in \resizebox
    resizebox_width: str | None = None
    array_stretch: float = 1
    # Set to e.g. "0.4\\textwidth" to use \wraptable{r}{width} instead of \table[h]
    wrap_width: str | None = None


def _apply_highlights(
    rows: list[tuple],
    values: list[list[float | None]],
    higher_is_better: bool,
    bold_best: bool,
    underline_second: bool,
) -> list[tuple]:
    n_data_cols = len(values[0])
    best: dict[int, int] = {}
    second: dict[int, int] = {}
    for j in range(n_data_cols):
        col = [(values[i][j], i) for i in range(len(rows)) if values[i][j] is not None]
        if not col:
            continue
        col.sort(key=lambda x: x[0], reverse=higher_is_better)
        best[j] = col[0][1]
        if len(col) > 1:
            second[j] = col[1][1]

    new_rows = []
    for i, row in enumerate(rows):
        cells = list(row)
        for j in range(n_data_cols):
            cell = str(cells[j + 1])
            SEP = "\\,$_{\\pm"
            if SEP in cell:
                before_pm, after_pm = cell.split(SEP, 1)
            else:
                before_pm, after_pm = cell, ""
            m = re.match(r"^(\\phantom\{0+\})+", before_pm)
            prefix = m.group(0) if m else ""
            inner = before_pm[len(prefix) :]
            if bold_best and best.get(j) == i:
                cell = (
                    f"{prefix}\\textbf{{{inner}}}{SEP}{after_pm}"
                    if after_pm
                    else f"{prefix}\\textbf{{{inner}}}"
                )
            elif underline_second and second.get(j) == i:
                cell = (
                    f"{prefix}\\underline{{{inner}}}{SEP}{after_pm}"
                    if after_pm
                    else f"{prefix}\\underline{{{inner}}}"
                )
            cells[j + 1] = cell
        new_rows.append(tuple(cells))
    return new_rows


def render_table(t: LatexTable, bold_best: bool = False, underline_second: bool = False) -> str:
    n_data_cols = len(t.headers) - 1

    col_format = t.col_format or ("l" + "c" * n_data_cols)
    headers_str = " & ".join(f"\\textbf{{{h}}}" for h in t.headers) + r" \\"

    rows = t.rows
    if t.values is not None and (bold_best or underline_second):
        rows = _apply_highlights(rows, t.values, t.higher_is_better, bold_best, underline_second)

    body_lines = []
    for i, row in enumerate(rows):
        body_lines.append("    " + " & ".join(str(c) for c in row) + r" \\")
        if t.midrule_after is not None and i in t.midrule_after:
            body_lines.append(r"    \midrule")

    tabular_lines = [
        f"  \\begin{{tabular}}{{{col_format}}}",
        r"    \toprule",
        f"    {headers_str}",
        r"    \midrule",
        *body_lines,
        r"    \bottomrule",
        r"  \end{tabular}",
    ]

    if t.resizebox_width:
        inner = "\n".join(tabular_lines)
        body_block = f"  \\resizebox{{{t.resizebox_width}}}{{!}}{{%\n{inner}%\n  }}"
    else:
        body_block = "\n".join(tabular_lines)

    if t.wrap_width:
        env_begin = f"\\begin{{wraptable}}{{r}}{{{t.wrap_width}}}"
        env_end = r"\end{wraptable}"
    else:
        env_begin = r"\begin{table}[h]"
        env_end = r"\end{table}"

    return "\n".join(
        [
            env_begin,
            r"  \centering",
            f"  \\caption{{{t.caption}}}",
            f"  \\label{{{t.label}}}",
            f"  \\renewcommand{{\\arraystretch}}{{{t.array_stretch}}}",
            body_block,
            env_end,
        ]
    )


def get_agg_table(
    df: pd.DataFrame,
    metric: str = "eval_score",
    bold_best: bool = False,
    underline_second: bool = False,
    higher_is_better: bool = True,
    optimizer_order: list[str] | None = None,
) -> pd.DataFrame:
    scale = 100 if "score" in metric else 1

    subset = df[df[metric].notna()][
        ["experiment_path", metric, "experiment.dataset", "experiment.optimizer"]
    ].drop_duplicates(subset="experiment_path")

    agg = (
        subset.groupby(["experiment.dataset", "experiment.optimizer"])[metric]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    mean_pivot = agg.pivot(
        index="experiment.optimizer", columns="experiment.dataset", values="mean"
    )
    std_pivot = agg.pivot(index="experiment.optimizer", columns="experiment.dataset", values="std")

    if optimizer_order is not None:
        order_lower = [o.lower() for o in optimizer_order]
        mean_pivot = mean_pivot.reindex(order_lower)
        std_pivot = std_pivot.reindex(order_lower)

    df_pretty = pd.DataFrame(index=mean_pivot.index)
    for col in mean_pivot.columns:
        m = (mean_pivot[col] * scale).round(2)
        s = (std_pivot[col] * scale).round(2)
        df_pretty[col] = m.astype(str) + " $\\pm$ " + s.astype(str)

    if bold_best or underline_second:
        numeric = mean_pivot * scale
        for col in df_pretty.columns:
            col_vals = numeric[col].dropna()
            if col_vals.empty:
                continue
            sorted_idx = col_vals.sort_values(ascending=not higher_is_better).index
            if bold_best and len(sorted_idx) >= 1:
                idx = sorted_idx[0]
                df_pretty.at[idx, col] = f"\\textbf{{{df_pretty.at[idx, col]}}}"
            if underline_second and len(sorted_idx) >= 2:
                idx = sorted_idx[1]
                df_pretty.at[idx, col] = f"\\underline{{{df_pretty.at[idx, col]}}}"

    return df_pretty
