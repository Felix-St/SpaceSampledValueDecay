"""Autorank CD diagram (left) next to the paired Base vs. SsVD plot (right).

Both panels use the scores from rank_seosnsrl.load_table(), so they match each other and
the tables written by rank_seosnsrl.py exactly. Run from the repo root.
"""
import contextlib
import io
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from autorank import autorank, plot_stats
from matplotlib.lines import Line2D
from scipy.stats import binomtest, t as t_dist

import rank_seosnsrl as rk  # also sets the shared matplotlib style

# ----------------------------------------------------------------------------- config
CD_WIDTH = 8.75              
PAIR_AXES = (6.25, 4.5)     # paired-plot axes (width, height) in inches, labels excluded   
                             
CITATIONS = "below"          # "below": citation in smaller text under the method name,
                             # "drop": remove it, "inline": keep autorank's label as is
CITE_SCALE = 0.75            # citation text size relative to FONTSIZE
    
GAP = 0.6                    # inches between the two panels (labels included)
FONTSIZE = 40                # one text size for both panels (rank_seosnsrl.py uses 24)
TEXTWIDTH_IN = 5.5           # LaTeX \textwidth, only used to report the printed font size
ERRORBARS = None #"se"             # paired plot, per marker over seeds: "se" (+-1 standard error),
                             # "ci95" (95% t-interval) or None
F_COL = None                 # method name of the SsVD variant; None = auto-detect
CD_LABELS = {}               # method -> label in the CD diagram
ENV_LABELS = {"MultiPadLunarLander (discr.)": "MPLL (discr.)", "MultiPadLunarLander (cont.)": "MPLL (cont.)"}    

OUT_FILE = os.path.join(rk.OUT_DIR, "cd_and_paired.pdf")

COLOR = "#185FA5"            # paired-plot markers
LINE_COLOR = "#888780"       # paired-plot connectors
MUTED = "#6b6b6b"            # base-algorithm labels and gains


# ----------------------------------------------------------------------------- CD diagram
def cd_result(table):
    """Friedman + Nemenyi on the same block of methods/envs as rank_seosnsrl.py."""
    block = rk.best_block(table)
    if block is None or block.shape[1] < 3:
        raise ValueError(f"Need >= 3 methods that share >= {rk.MIN_ENVS} envs for a CD diagram")
    block = block.rename(columns=CD_LABELS)
    with contextlib.redirect_stdout(io.StringIO()):  # autorank prints pooled raw-score stats
        result = autorank(block, alpha=rk.ALPHA, order="descending", force_mode="nonparametric")
    print(f"CD diagram: {block.shape[1]} methods x {block.shape[0]} envs, "
          f"Friedman {rk.fmt_p(result.pvalue)}, critical distance = {result.cd:.2f}")
    return result


def draw_cd(ax, result):
    """rank_seosnsrl.plot_cd, but drawn into an existing axes instead of its own figure."""
    # width = axes width in inches, so autorank's 1-unit label margins are real inches
    plot_stats(result, allow_insignificant=True, ax=ax, width=CD_WIDTH)
    if not rk.CD_SHOW_RULER:
        # autorank draws in axes coordinates with an inverted y axis; the ruler ("CD" label
        # plus a bar with two end ticks) is the only content above the rank-axis labels
        label_y = min(t.get_position()[1] for t in ax.texts if t.get_text() != "CD")
        for text in [t for t in ax.texts if t.get_text() == "CD"]:
            text.remove()
        for line in [l for l in ax.lines if max(l.get_ydata()) < label_y]:
            line.remove()
        ys = [l.get_ydata() for l in ax.lines if l.get_color() != "w"]  # skip sizing helper
        ax.set_ylim(max(map(max, ys)) + 0.02, label_y - 0.04)
    for text in ax.texts:
        text.set_fontsize(FONTSIZE)
    for line in ax.lines:
        line.set_linewidth(line.get_linewidth() * rk.CD_LINE_SCALE)
    if CITATIONS != "inline":
        split_citations(ax)


def split_citations(ax):
    """'Method (Author et al, 2022)' -> 'Method', with the citation in smaller gray text
    underneath (CITATIONS = "below") or removed (CITATIONS = "drop")."""
    for text in list(ax.texts):
        m = re.fullmatch(r"(.*?)\s*\(([^()]*\d{4}[^()]*)\)", text.get_text())
        if not m:
            continue
        text.set_text(m.group(1))
        if CITATIONS == "below":
            right_aligned = text.get_ha() == "right"  # labels left of the scale
            ax.annotate(f"({m.group(2)})", xy=(1 if right_aligned else 0, 0), xycoords=text,
                        xytext=(0, -1), textcoords="offset points", ha=text.get_ha(),
                        va="top", fontsize=FONTSIZE * CITE_SCALE, color=MUTED)


# ----------------------------------------------------------------------------- per-seed scores
def seed_score(ts, rs, max_t, bin_size):
    """Mean online return of one seed: average episode return per fixed-width time bin,
    then the mean over all bins that contain an episode (as for the seed-averaged curve)."""
    ts, rs = np.asarray(ts, dtype=float), np.asarray(rs, dtype=float)
    if ts.shape != rs.shape or ts.ndim != 1:
        raise ValueError(f"Expected one list of timesteps and one of returns per seed, got "
                         f"shapes {ts.shape} and {rs.shape}")
    n_bins = int(np.ceil(max_t / bin_size))
    idx = (ts // bin_size).astype(int)
    keep = (idx >= 0) & (idx < n_bins) & np.isfinite(rs)
    sums = np.bincount(idx[keep], weights=rs[keep], minlength=n_bins)
    counts = np.bincount(idx[keep], minlength=n_bins)
    if not counts.any():
        return np.nan
    filled = counts > 0
    return np.mean(sums[filled] / counts[filled])


def per_seed_scores():
    """Mean online return of every seed, computed like rank_seosnsrl.load_table (same env
    labels, bin size, exclusions and method mapping) but per seed instead of on the
    seed-averaged curve. Returns {(env label, method): array of per-seed scores}."""
    import gymnasium as gym
    from utils.plotting import model_to_simplified

    runs = []  # (env label, legend name, key, per-seed scores)
    for fname in rk.FNAMES:
        with open(os.path.join("results", fname + ".json")) as f:
            data = json.load(f)
        env_str, models = data["environment"], data["models"]
        backbones = [b for b in rk.BACKBONES if b in str(models)]
        label = env_str
        if env_str == "MultiPadLunarLander":
            label += " (cont.)" if "SAC" in backbones else " (discr.)" if "DQN" in backbones else ""
        label = rk.ENV_LABEL_OVERRIDE.get(fname, label)

        env = gym.make(env_str)

        try:
            bin_size = 5 * env._max_episode_steps
        except:
            # TEST FOR ATARI, CHANGE TO LARGE NUMBER and probably multipad too
            try:
                bin_size = 5 * env.env.unwrapped.max_steps
            except:
                raise Exception("Unknown max steps")
            
        env.close()

        for key in models:
            name = model_to_simplified.get(key[:-3], key[:-3])
            if {name, key, (fname, key)} & set(rk.EXCLUDE):
                continue
            scores = [seed_score(ts, rs, data["total_timesteps"], bin_size)
                      for ts, rs in zip(data[key]["recorded_timesteps"],
                                        data[key]["reward_history"])]
            runs.append((label, name, key, np.array(scores)))

    with contextlib.redirect_stdout(io.StringIO()):  # load_table already printed the mapping
        mapping = rk.family_map([(name, key) for _, name, key, _ in runs])
    mapping = {k: rk.RENAME.get(v, v) for k, v in mapping.items()}
    return {(label, mapping[name]): scores for label, name, _, scores in runs}


def error_halfwidth(scores):
    """Half-width of the error bar for the mean of per-seed scores (seeds without any
    finished episode are skipped)."""
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if ERRORBARS is None or n < 2:
        return 0.0
    se = np.std(scores, ddof=1) / np.sqrt(n)
    return se * (t_dist.ppf(0.975, n - 1) if ERRORBARS == "ci95" else 1.0)


# ----------------------------------------------------------------------------- paired plot
def find_f_col(table):
    if F_COL is not None:
        return F_COL
    cands = [c for c in table.columns if re.search(r"SsVD|(?<![A-Za-z])F(?![A-Za-z])", c)]
    if len(cands) != 1:
        raise ValueError(f"Cannot identify the SsVD method among {list(table.columns)}; set F_COL")
    return cands[0]


def env_label(env):
    return ENV_LABELS.get(env, re.sub(r"(Env)?-v\d+(?=\s|$)", "", env))


def paired_rows(table, backbone, seeds):
    """(label, backbone, base, F, base error, F error) per env, largest gain first, all
    min-max normalized over the methods that ran in that env (0 = worst, 1 = best)."""
    f_col, rows, wins = find_f_col(table), [], 0
    for env, row in table.iterrows():
        base, f = row[rk.BASE_NAME], row[f_col]
        if np.isnan(base) or np.isnan(f):
            print(f"  paired plot: skipping {env} (base or F missing)")
            continue
        lo, hi = row.min(), row.max()
        errs = []
        for method, mean in ((rk.BASE_NAME, base), (f_col, f)):
            scores = seeds[(env, method)]
            n_ok = int(np.isfinite(scores).sum())
            if n_ok < len(scores):
                print(f"  {env}, {method}: {len(scores) - n_ok} of {len(scores)} seeds "
                      f"have no finished episode and are skipped")
            if n_ok and abs(np.nanmean(scores) - mean) > 0.02 * (hi - lo):  # should match
                print(f"  WARNING: {env}, {method}: per-seed mean {np.nanmean(scores):.2f} "
                      f"differs from the table score {mean:.2f}")
            errs.append(error_halfwidth(scores) / (hi - lo))
        wins += f > base
        rows.append((env_label(env), backbone[env], (base - lo) / (hi - lo),
                     (f - lo) / (hi - lo), *errs))
        if ERRORBARS:
            print(f"  {env}: base {rows[-1][2]:.2f} +- {errs[0]:.2f}, "
                  f"F {rows[-1][3]:.2f} +- {errs[1]:.2f} ({ERRORBARS}, normalized)")
    p = binomtest(int(wins), len(rows), 0.5).pvalue
    print(f"Paired plot: {f_col} better than {rk.BASE_NAME} in {wins}/{len(rows)} envs, "
          f"sign test {rk.fmt_p(p)}")
    return sorted(rows, key=lambda r: r[3] - r[2], reverse=True)


def draw_paired(ax, rows):
    k = FONTSIZE / 24  # markers and lines grow with the text (tuned at FONTSIZE = 24)
    y = np.arange(len(rows))[::-1]
    for yi, (env, algo, base, f, err_base, err_f) in zip(y, rows):
        ax.plot([base, f], [yi, yi], color=LINE_COLOR, lw=3 * k, zorder=1)
        for x, err in ((base, err_base), (f, err_f)):
            if err > 0:  # drawn below the markers, so only the whiskers show
                ax.errorbar(x, yi, xerr=err, fmt="none", ecolor=COLOR, elinewidth=1.5 * k,
                            capsize=6 * k, capthick=1.5 * k, zorder=1.5)
        ax.scatter(base, yi, s=180 * k**2, facecolors="white", edgecolors=COLOR,
                   linewidths=3 * k, zorder=2)
        ax.scatter(f, yi, s=200 * k**2, color=COLOR, zorder=3)
        ax.text(1.03, yi, f"{f - base:+.2f}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", color=MUTED, fontsize=FONTSIZE)
        algo_txt = ax.text(-0.02, yi, algo, transform=ax.get_yaxis_transform(),
                           va="center", ha="right", color=MUTED, fontsize=FONTSIZE)
        ax.annotate(env, xy=(0, 0), xycoords=algo_txt, xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="bottom", fontsize=FONTSIZE)
    ax.text(1.03, 1.0, r"$\Delta$", transform=ax.transAxes, va="bottom", ha="left",
            color=MUTED, fontsize=FONTSIZE)

    # widen the x range if an error bar reaches beyond [0, 1]
    x_lo = min([0] + [min(b - eb, f - ef) for _, _, b, f, eb, ef in rows])
    x_hi = max([1] + [max(b + eb, f + ef) for _, _, b, f, eb, ef in rows])
    ax.set_xlim(x_lo - 0.03, x_hi + 0.03)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="x", labelsize=FONTSIZE)
    ax.set_yticks([])
    ax.set_xlabel("Normalized mean online return", fontsize=FONTSIZE)
    ax.grid(False)
    ax.grid(axis="x", color="#d3d1c7", lw=1.2 * k)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    handles = [
        Line2D([], [], marker="o", ls="", markersize=14 * k, markerfacecolor="white",
               markeredgecolor=COLOR, markeredgewidth=3 * k, label="Base (DQN / SAC)"),
        Line2D([], [], marker="o", ls="", markersize=14 * k, color=COLOR, label="+ SsVD (-F)"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, fontsize=FONTSIZE, handletextpad=0.3, columnspacing=1.5,
              borderaxespad=1.0)  # sits above the Delta header


# ----------------------------------------------------------------------------- layout
def overhang(ax, renderer):
    """Inches that ax's labels stick out of its axes box: (left, right, bottom, top).
    Needed because autorank places its labels outside the axes in axes coordinates, which
    breaks matplotlib's layout engines (they keep shrinking the axes)."""
    tb, ab, dpi = ax.get_tightbbox(renderer), ax.get_window_extent(renderer), ax.figure.dpi
    return ((ab.x0 - tb.x0) / dpi, (tb.x1 - ab.x1) / dpi, (ab.y0 - tb.y0) / dpi, (tb.y1 - ab.y1) / dpi)


def place_side_by_side(fig, ax_cd, ax_pair):
    """Fixed-width axes, figure sized so that all labels fit. The CD diagram is stretched
    vertically so that its content (rank numbers to lowest label) spans the same height as
    the paired plot's (legend to x label). Iterated because the CD labels' overhang changes
    slightly with the axes height."""
    cd_h = PAIR_AXES[1]  # first guess
    for _ in range(6):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        cl, cr, cb, ct = overhang(ax_cd, renderer)
        pl, pr, pb, pt = overhang(ax_pair, renderer)
        height = pb + PAIR_AXES[1] + pt
        cd_h = max(height - cb - ct, 0.5)
        width = cl + CD_WIDTH + cr + GAP + pl + PAIR_AXES[0] + pr
        fig.set_size_inches(width, height)
        ax_cd.set_position([cl / width, cb / height, CD_WIDTH / width, cd_h / height])
        x_pair = cl + CD_WIDTH + cr + GAP + pl
        ax_pair.set_position([x_pair / width, pb / height,
                              PAIR_AXES[0] / width, PAIR_AXES[1] / height])
    return width


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(rk.OUT_DIR, exist_ok=True)
    table, backbone = rk.load_table()

    fig = plt.figure()
    ax_cd, ax_pair = fig.add_axes([0, 0, 0.4, 0.5]), fig.add_axes([0.5, 0, 0.4, 0.5])
    draw_cd(ax_cd, cd_result(table))
    draw_paired(ax_pair, paired_rows(table, backbone, per_seed_scores()))
    width = place_side_by_side(fig, ax_cd, ax_pair)

    fig.savefig(OUT_FILE, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved {OUT_FILE}: {width:.1f} in wide; at \\textwidth = {TEXTWIDTH_IN} in, "
          f"text prints at ~{FONTSIZE * TEXTWIDTH_IN / width:.1f} pt")
   


if __name__ == "__main__":
    main()