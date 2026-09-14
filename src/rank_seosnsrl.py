"""
Global ranking of approaches across environments with autorank.

    1. Each (env, method) learning curve -> one score: the mean of the binned episode-return
    curve over all of training (same binning as plot_seosnsrl.py).
    2. Backbone variants are merged into one method ("X-DQN" + "X-SAC" -> "X").
    3. Methods are ranked within each env, and the ranks are aggregated over envs
"""

import contextlib
import io
import itertools
import json
import os
import re
import matplotlib.patheffects as pe

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autorank import autorank, plot_stats
from scipy.stats import binomtest, friedmanchisquare

plt.rc("font", **{"family": "serif", "serif": ["times"]})
plt.rc("text", usetex=True)
plt.rc("axes", titlesize=26)
plt.rc("axes", labelsize=24)
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.axisbelow"] = True
matplotlib.rc("xtick", labelsize=24)
matplotlib.rc("ytick", labelsize=24)

# ----------------------------------------------------------------------------- config
pd.set_option("display.max_columns", None, "display.max_rows", None, "display.width", 250)


FNAMES = [
    "CartPole-v1",
    "MountainCar-v0",
    "ChooseBoxEnv-v0",
    "InvertedPendulum-v5",
    "MultiPadLunarLanderDiscrete",
    "MultiPadLunarLander",
]
OUT_DIR = "./src/imgs/ranking"

ALPHA = 0.05
MIN_ENVS = 5             # autorank needs >= 5 complete envs
MIN_SHARED_TEST = 6      # with fewer non-tied envs, the sign test cannot reach p < 0.05

BACKBONES = ["DQN", "SAC"]
BASE_NAME = "Base (DQN/SAC)"  # name for the plain backbones themselves
FAMILY_OVERRIDE = {           # legend name or raw model key -> method name
    "LimtiedDQN": "Limited",
    "LimtiedSAC": "Limited",
}
RENAME = {               # method name -> display name, applied after the merging above
    "F": "Base + SsVD (-F)",
}
EXCLUDE = []             # legend names, raw keys or (fname, key) tuples to skip
ENV_LABEL_OVERRIDE = {}  # fname -> row label, if two files get the same label
MUST_INCLUDE = []        # methods that must appear in the CD diagram
CD_FONTSIZE = 24         # CD diagram text size
CD_LINE_SCALE = 2.5      # CD diagram line thickness relative to autorank's default
CD_WIDTH = 10            # CD diagram width in inches
CD_HEIGHT_SCALE = 0.6
CD_SHOW_RULER = False    # draw the "CD" ruler above the axis (its length is in the caption)

# ----------------------------------------------------------------------------- loading
def _k(s):
    return re.sub(r"\s+", "", str(s)).lower()


def strip_backbone(name):
    """'X-DQN', 'X (SAC)', 'X$_{DQN}$' -> 'X';  'DQN' -> BASE_NAME;  'DDQN' unchanged."""
    fam = name
    for b in BACKBONES:
        fam = re.sub(rf"(?<![A-Za-z]){b}(?![A-Za-z])", " ", fam)
    for leftover in [r"\\(text|mathrm|textsc|textit)\{\s*\}", r"\(\s*\)", r"[_^]\{\s*\}", r"\$\s*\$"]:
        fam = re.sub(leftover, " ", fam)
    return re.sub(r"\s+", " ", fam).strip(" -_+/,:") or BASE_NAME


def family_map(runs):
    """runs: list of (legend name, raw key) -> dict legend name -> method name."""
    mapping, glued, used = {}, {}, set()
    for name, key in runs:
        match = [k for k in FAMILY_OVERRIDE if _k(k) in {_k(name), _k(key), _k(key[:-3])}]
        if match:
            used.add(match[0])
            mapping[name] = FAMILY_OVERRIDE[match[0]]
            continue
        mapping[name] = strip_backbone(name)
        if mapping[name] == name:  # backbone possibly glued on, e.g. "LimtiedDQN"
            for b in BACKBONES:
                m = re.search(b, name, flags=re.IGNORECASE)
                if m:
                    stem = (name[:m.start()] + name[m.end():]).strip(" -_+/,:")
                    glued.setdefault(stem, {}).setdefault(b, set()).add(name)
                    break
    # merge glued names only if the stem exists with >= 2 backbones ("DDQN" has no "DSAC")
    for stem, by_backbone in glued.items():
        if stem and len(by_backbone) >= 2:
            for names in by_backbone.values():
                mapping.update(dict.fromkeys(names, stem))
    for k in set(FAMILY_OVERRIDE) - used:
        print(f"WARNING: FAMILY_OVERRIDE entry {k!r} matched no run")
    return mapping


def load_table():
    """Returns the env x method score table (NaN = not run) and each env's backbone."""
    import ale_py
    import gymnasium as gym
    from environments import nonstationary_lunar_lander, river_swim, choose_box  # noqa: F401 (registers envs)
    from utils.plotting import bin_multiple_seeds, model_to_simplified

    gym.register_envs(ale_py)
    rows, env_backbone = [], {}
    for fname in FNAMES:
        with open(os.path.join("results", fname + ".json")) as f:
            data = json.load(f)
        env_str, models = data["environment"], data["models"]
        backbones = [b for b in BACKBONES if b in str(models)]

        label = env_str
        if env_str == "MultiPadLunarLander":  # same labels as in the plots
            label += " (cont.)" if "SAC" in backbones else " (discr.)" if "DQN" in backbones else ""
        label = ENV_LABEL_OVERRIDE.get(fname, label)
        env_backbone[label] = "/".join(backbones) or "other"

        env = gym.make(env_str)  # same bin size as in the plots
        try:
            bin_size = 3 * env._max_episode_steps
        except Exception:
            bin_size = 500
        env.close()

        for key in models:
            name = model_to_simplified.get(key[:-3], key[:-3])
            if {name, key, (fname, key)} & set(EXCLUDE):
                continue
            curve = bin_multiple_seeds(data[key]["recorded_timesteps"], data[key]["reward_history"],
                                       max_t=data["total_timesteps"], bin_size=bin_size)["mean"]
            rows.append({"env": label, "legend": name, "key": key, "file": fname,
                         "score": np.nanmean(curve)})

    df = pd.DataFrame(rows)
    mapping = family_map(list(zip(df["legend"], df["key"])))
    for m in set(RENAME) - set(mapping.values()):
        print(f"WARNING: RENAME entry {m!r} matched no method")
    mapping = {k: RENAME.get(v, v) for k, v in mapping.items()}
    df["method"] = df["legend"].map(mapping)
    print("Legend name -> method:")
    for k, v in sorted(mapping.items()):
        print(f"  {k:30s} -> {v}")

    dups = df[df.duplicated(["env", "method"], keep=False)]
    if len(dups):
        raise ValueError(
            "Several runs ended up in the same (env, method) cell. Fix via ENV_LABEL_OVERRIDE "
            "(different files), EXCLUDE (same file) or FAMILY_OVERRIDE (wrong merge):\n"
            + dups.sort_values(["env", "method"])[["env", "method", "file", "key", "legend"]].to_string(index=False))
    table = df.pivot(index="env", columns="method", values="score")
    return table, pd.Series(env_backbone).reindex(table.index)


# ----------------------------------------------------------------------------- statistics
def within_env_ranks(table):
    """Rank 1 = best in each env, plus the rank scaled to [0, 1] (0 = best) so that envs
    with different numbers of methods are comparable."""
    ranks = table.rank(axis=1, ascending=False)
    k = table.notna().sum(axis=1)
    return ranks, (ranks - 1).div((k - 1).where(k > 1), axis=0)


def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    order = np.flatnonzero(~np.isnan(p))
    order = order[np.argsort(p[order])]
    adjusted = np.maximum.accumulate((len(order) - np.arange(len(order))) * p[order])
    out[order] = np.minimum(adjusted, 1)
    return out


def pairwise(table):
    """Wins/ties/losses on the envs both methods ran on, sign test with Holm correction.
    The sign test only uses who won in each env, so reward scales don't matter
    (a Wilcoxon test would rank the raw differences, letting large-reward envs dominate)."""
    rows = []
    for a, b in itertools.combinations(table.columns, 2):
        d = (table[a] - table[b]).dropna()
        wins, losses = int((d > 0).sum()), int((d < 0).sum())
        p = binomtest(wins, wins + losses).pvalue if wins + losses >= MIN_SHARED_TEST else np.nan
        rows.append({"A": a, "B": b, "n_shared": len(d), "A_wins": wins,
                     "ties": len(d) - wins - losses, "B_wins": losses, "p": p})
    res = pd.DataFrame(rows).set_index(["A", "B"])
    res["p_holm"] = holm(res["p"])
    return res


def best_block(table):
    """Largest set of methods (containing MUST_INCLUDE) that all ran on the same
    >= MIN_ENVS envs; ties are broken by the number of envs."""
    missing = set(MUST_INCLUDE) - set(table.columns)
    if missing:
        raise ValueError(f"MUST_INCLUDE {missing} not among the methods {list(table.columns)}")
    avail = table.notna()
    others = [m for m in table.columns if m not in MUST_INCLUDE]
    for r in range(len(others), -1, -1):
        blocks = [list(MUST_INCLUDE) + list(s) for s in itertools.combinations(others, r)]
        blocks = [(avail[m].all(axis=1).sum(), m) for m in blocks if len(m) >= 2]
        blocks = [b for b in blocks if b[0] >= MIN_ENVS]
        if blocks:
            methods = max(blocks, key=lambda b: b[0])[1]
            return table.loc[avail[methods].all(axis=1), methods]
    return None


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def run_autorank(block):
    """Friedman test + Nemenyi critical distance on the within-env ranks. Only rank-based
    results are reported: autorank's own report and LaTeX table also list medians, MADs and
    effect sizes of the raw scores pooled over envs, which mix reward scales."""
    n, k = block.shape
    if k == 2:  # autorank would use a Wilcoxon test on raw differences; use the sign test instead
        a, b = block.columns
        wins, losses = int((block[a] > block[b]).sum()), int((block[a] < block[b]).sum())
        p = binomtest(wins, wins + losses).pvalue if wins + losses else np.nan
        print(f"Sign test on {n} envs: {a} wins {wins}, {b} wins {losses}, {fmt_p(p)}")
        return

    with contextlib.redirect_stdout(io.StringIO()):  # autorank prints the pooled statistics
        # nonparametric = rank-based, since reward scales differ between envs
        result = autorank(block, alpha=ALPHA, order="descending", force_mode="nonparametric")
    chi2 = friedmanchisquare(*block.T.to_numpy()).statistic
    mr = result.rankdf["meanrank"].sort_values()
    significant = result.pvalue < ALPHA

    print(f"Friedman test: chi2({k - 1}) = {chi2:.2f}, {fmt_p(result.pvalue)} "
          f"({'significant' if significant else 'not significant'} at alpha = {ALPHA})")
    print(f"Nemenyi critical distance: {result.cd:.3f}")
    print("Mean rank (1 = best):\n" + mr.round(2).to_string())
    if significant:  # post-hoc comparisons only after a significant omnibus test
        pairs = [f"{a} vs {b}" for a, b in itertools.combinations(mr.index, 2)
                 if abs(mr[a] - mr[b]) > result.cd]
        print("Significantly different pairs: " + (", ".join(pairs) or "none"))

    tex = "\n".join([
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{Mean rank over the {n} environments on which all methods were evaluated "
        rf"(1 = best). Friedman test: $\chi^2({k - 1}) = {chi2:.2f}$, ${fmt_p(result.pvalue)}$; "
        rf"Nemenyi critical distance: {result.cd:.2f}.}}",
        r"\label{tab:mean_ranks}", r"\begin{tabular}{lr}", r"\toprule",
        r"Method & Mean rank \\", r"\midrule",
        *[rf"{m} & {r:.2f} \\" for m, r in mr.items()],
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])
    with open(os.path.join(OUT_DIR, "mean_ranks.tex"), "w") as f:
        f.write(tex + "\n")
    print("\n" + tex)
    plot_cd(result).savefig(os.path.join(OUT_DIR, "cd_diagram.pdf"), bbox_inches="tight")


def plot_cd(result):
    """autorank's CD diagram, enlarged: autorank uses the default font size and thin lines
    and has no option for either, so they are changed after drawing."""
    ax = plot_stats(result, allow_insignificant=True, width=CD_WIDTH + 2)
    kept = 1.0  # fraction of the drawn height that is still used after dropping the ruler
    if not CD_SHOW_RULER:
        # autorank draws in axes coordinates with an inverted y axis: the ruler ("CD" label
        # plus a bar with two end ticks) is the only content above the rank-axis labels
        label_y = min(t.get_position()[1] for t in ax.texts if t.get_text() != "CD")
        for text in [t for t in ax.texts if t.get_text() == "CD"]:
            text.remove()
        for line in [l for l in ax.lines if max(l.get_ydata()) < label_y]:
            line.remove()
        ys = [l.get_ydata() for l in ax.lines if l.get_color() != "w"]  # skip the sizing helper
        bottom, top = max(map(max, ys)) + 0.02, label_y - 0.04
        ax.set_ylim(bottom, top)
        kept = bottom - top  # autorank's y axis runs 1 -> 0, so this is the remaining share
    for text in ax.texts:
        text.set_fontsize(CD_FONTSIZE)
    for line in ax.lines:
        line.set_linewidth(line.get_linewidth() * CD_LINE_SCALE)
    # autorank sizes the height for ~10pt text and a full y axis: stretch it for larger text,
    # and shrink it by `kept` so dropping the ruler does not stretch the rest of the diagram
    fig = ax.figure
    fig.set_size_inches(CD_WIDTH, fig.get_figheight() * CD_FONTSIZE / 10 * CD_HEIGHT_SCALE * kept)

    return fig


# ----------------------------------------------------------------------------- plots
def plot_heatmap(ranks, norm):
    """Within-env rank of every method; blank = not run."""
    order = norm.mean().sort_values().index
    r, n = ranks[order], norm[order]
    fig, ax = plt.subplots(figsize=(14,10))
    im = ax.imshow(np.ma.masked_invalid(r.to_numpy()), cmap="inferno",  aspect="auto", alpha = 0.85)
    for (i, j), v in np.ndenumerate(r.to_numpy()):
        if not np.isnan(v):
            ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=20, color="w", path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
    ax.set_xticks(range(len(order)), order, rotation=45, ha="right")

    ax.set_yticks(range(len(r)), r.index)
    ax.grid(False)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rank_heatmap.pdf"), bbox_inches="tight")


def plot_norm_ranks(norm, n_boot=10000):
    """Mean normalized rank of every method with a bootstrap 95% CI over its envs.
    Descriptive only: methods ran on different envs against different opponents."""
    rng = np.random.default_rng(0)
    stats = []
    for m in norm.columns:
        v = norm[m].dropna().to_numpy()
        lo, hi = np.percentile(rng.choice(v, (n_boot, len(v))).mean(axis=1), [2.5, 97.5])
        stats.append((m, v.mean(), lo, hi, len(v)))
    st = pd.DataFrame(stats, columns=["method", "mean", "lo", "hi", "n"]).sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 0.8 * len(st) + 1.5))
    y = np.arange(len(st))
    ax.errorbar(st["mean"], y, xerr=[st["mean"] - st["lo"], st["hi"] - st["mean"]],
                fmt="o", capsize=4, color="k", markersize=8)
    ax.set_yticks(y, [f"{m}  (n={n})" for m, n in zip(st["method"], st["n"])])
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("mean normalized rank (0 = best)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "norm_rank_all.pdf"), bbox_inches="tight")


def scores_to_latex(table, ranks):
    """Score table as LaTeX: one row per env, the within-env rank in parentheses, best in
    bold. Scores are only comparable within a row, since reward scales differ between envs."""
    body = []
    for env, row in table.iterrows():
        cells = []
        for method in table.columns:
            v, r = row[method], ranks.at[env, method]
            cell = "--" if pd.isna(v) else f"{v:,.1f} ({r:.0f})"
            cells.append(rf"\textbf{{{cell}}}" if r == 1 else cell)
        body.append(f"{env} & " + " & ".join(cells) + r" \\")

    return "\n".join([
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Mean online return per environment, averaged over seeds, with the "
        r"within-environment rank in parentheses (best in bold). Returns are only comparable "
        r"within a row.}",
        r"\label{tab_app:mor}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l" + "r" * len(table.columns) + "}", r"\toprule",
        "Environment & " + " & ".join(table.columns) + r" \\", r"\midrule",
        *body,
        r"\bottomrule", r"\end{tabular}",  r"}" r"\end{table}",
    ])


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    table, backbone = load_table()
    ranks, norm = within_env_ranks(table)

    summary = pd.DataFrame({"n_envs": table.notna().sum(), "mean_norm_rank": norm.mean(),
                            "n_first": (ranks == 1).sum()})
    if backbone.nunique() > 1:  # does the merged order hold within each backbone group?
        for g, envs in backbone.groupby(backbone).groups.items():
            summary[f"{g} envs (n={len(envs)})"] = norm.loc[envs].mean()
    summary = summary.sort_values("mean_norm_rank")
    pw = pairwise(table)

    for name, df in [("scores", table), ("rank_summary", summary), ("pairwise", pw)]:
        df.to_csv(os.path.join(OUT_DIR, f"{name}.csv"))
    tex = scores_to_latex(table, ranks)
    with open(os.path.join(OUT_DIR, "scores.tex"), "w") as f:
        f.write(tex + "\n")
    print("\nScores (NaN = not run):\n", table.round(2))
    print("\n" + tex)
    print("\nMean normalized rank (0 = best):\n", summary.round(3))
    print("\nPairwise comparisons on shared envs:\n", pw.round(4))

    block = best_block(table)
    if block is None:
        print(f"\nNo {MIN_ENVS} envs on which >= 2 methods all ran: autorank skipped.")
    else:
        print(f"\nautorank: {block.shape[1]} methods x {block.shape[0]} envs {list(block.index)}")
        for m in table.columns.difference(block.columns):
            n_left = len(table[list(block.columns) + [m]].dropna())
            print(f"  not in CD diagram: {m} (ran on {table[m].notna().sum()} envs; "
                  f"adding it leaves {n_left} complete envs)")
        run_autorank(block)

    plot_heatmap(ranks.rename(index=lambda e: f"{e}"),
                 norm.rename(index=lambda e: f"{e} [{backbone[e]}]"))
    plot_norm_ranks(norm)
    plt.show()


if __name__ == "__main__":
    main()