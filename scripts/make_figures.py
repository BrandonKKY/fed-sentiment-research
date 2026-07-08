"""Report figures (light-mode PNGs under output/figures/).

Colors and chart chrome follow a validated reference palette (categorical
slots assigned in fixed order; recessive grid/axes; direct labels instead of
legends where a series color is low-contrast).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "output" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
S1_BLUE = "#2a78d6"   # categorical slot 1
S2_AQUA = "#1baf7a"   # categorical slot 2 (sub-3:1 on light: direct-label it)

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titlecolor": INK,
    }
)

TRAIN_END = pd.Timestamp("2020-12-31")
SAMPLE_START = pd.Timestamp("2015-01-01")


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig_timeseries(scores):
    fig, ax = plt.subplots(figsize=(9.2, 3.6), dpi=150)
    style_axes(ax)
    for doc_type, color, label, dodge in (
        ("statement", S1_BLUE, "Statements", -11),
        ("minutes", S2_AQUA, "Minutes", 11),
    ):
        sub = scores[scores.doc_type == doc_type].sort_values("avail_date")
        ax.plot(
            sub["avail_date"], sub["lex_abg_fomc_net"],
            color=color, linewidth=2, solid_capstyle="round",
        )
        last = sub.iloc[-1]
        ax.annotate(
            label,
            (last["avail_date"], last["lex_abg_fomc_net"]),
            xytext=(8, dodge), textcoords="offset points",
            color=color, fontsize=9.5, fontweight="bold", va="center",
        )
    ax.axhline(0, color=AXIS, linewidth=1)
    for x, label in ((SAMPLE_START, "sample start"), (TRAIN_END, "train | test")):
        ax.axvline(x, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.annotate(
            label, (x, 0.955), xycoords=("data", "axes fraction"),
            xytext=(5, 0), textcoords="offset points",
            color=MUTED, fontsize=8.5, ha="left",
        )
    ax.annotate(
        "2006–2014: embedding pretrain corpus only",
        (pd.Timestamp("2010-06-01"), -0.93), color=MUTED, fontsize=8.5, ha="center",
    )
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Net hawkishness (extended lexicon)")
    ax.set_title("Hawkish–dovish sentiment of FOMC communications, 2006–2026")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sentiment_timeseries.png", bbox_inches="tight")
    plt.close(fig)


def fig_announcement_scatter(events):
    events = events.copy()
    events["dgs2_ann_bp"] = events["DGS2_ann"] * 100
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=150, sharex=True, sharey=True)
    splits = (
        ("Train 2015–2020", events[events.avail_date <= TRAIN_END]),
        ("Test 2021–2026", events[events.avail_date > TRAIN_END]),
    )
    for ax, (title, sub) in zip(axes, splits):
        style_axes(ax)
        sub = sub.dropna(subset=["lex_abg_fomc_net", "dgs2_ann_bp"])
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.axvline(0, color=GRID, linewidth=0.8)
        ax.scatter(
            sub["lex_abg_fomc_net"], sub["dgs2_ann_bp"],
            s=42, color=S1_BLUE, alpha=0.75, edgecolors=SURFACE, linewidths=1,
        )
        r = sub["lex_abg_fomc_net"].corr(sub["dgs2_ann_bp"])
        fit = pd.Series(
            [sub["dgs2_ann_bp"].mean() + r * sub["dgs2_ann_bp"].std()
             / sub["lex_abg_fomc_net"].std() * (x - sub["lex_abg_fomc_net"].mean())
             for x in (-0.9, 0.9)],
            index=[-0.9, 0.9],
        )
        ax.plot(fit.index, fit.values, color=INK2, linewidth=1.2, linestyle=(0, (5, 4)))
        ax.set_title(f"{title}   (n={len(sub)}, r={r:+.2f})")
        ax.set_xlabel("Statement net hawkishness (extended lexicon)")
    axes[0].set_ylabel("Same-day 2y Treasury yield change (bp)")
    fig.suptitle(
        "Hawkish statements coincide with same-day yield rises — contemporaneous, not predictive",
        fontsize=11, color=INK, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "announcement_day_scatter.png", bbox_inches="tight")
    plt.close(fig)


def fig_replication(grid):
    fwd = grid[grid.outcome != "ann"].copy()
    train = fwd[fwd.split == "train"].copy()
    train["abs_r"] = train.pearson_r.abs()
    top = train.nlargest(5, "abs_r")

    rows = []
    for _, r in top.iterrows():
        twin = fwd[
            (fwd.split == "test") & (fwd.doc_type == r.doc_type)
            & (fwd.signal == r.signal) & (fwd.asset == r.asset)
            & (fwd.outcome == r.outcome)
        ].iloc[0]
        label = f"{r.doc_type} · {r.signal} → {r.asset} {r.outcome.replace('fwd_', '+')}d"
        rows.append((label, r.pearson_r, twin.pearson_r))
    # the pre-registered primary hypothesis, for honesty
    prim = fwd[(fwd.doc_type == "statement") & (fwd.signal == "lex_abg_net_chg")
               & (fwd.asset == "DGS2") & (fwd.outcome == "fwd_5")]
    rows.append((
        "PRIMARY: statement · lex_abg_net_chg → DGS2 +5d",
        prim[prim.split == "train"].iloc[0].pearson_r,
        prim[prim.split == "test"].iloc[0].pearson_r,
    ))

    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(9.6, 3.4), dpi=150)
    style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=AXIS, linewidth=1)
    ys = list(range(len(rows), 0, -1))
    for y, (label, tr, te) in zip(ys, rows):
        ax.plot([tr, te], [y, y], color=GRID, linewidth=1.6, zorder=1)
        ax.scatter([tr], [y], s=64, color=S1_BLUE, zorder=2)
        ax.scatter([te], [y], s=64, color=S2_AQUA, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _, _ in rows], fontsize=8.8, color=INK2)
    ax.set_xlim(-0.65, 0.65)
    ax.set_ylim(0.4, len(rows) + 0.6)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=S1_BLUE, label="train"),
            Line2D([], [], marker="o", linestyle="", color=S2_AQUA, label="test"),
        ],
        loc="upper left", frameon=False, fontsize=9, handletextpad=0.2,
    )
    ax.set_xlabel("Pearson r of forward market move on sentiment signal")
    ax.set_title("Top-5 in-sample correlations do not replicate out of sample")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "replication_dumbbell.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(
        ROOT / "data" / "processed" / "scores.csv", parse_dates=["meeting_date", "avail_date"]
    )
    events = pd.read_csv(
        ROOT / "output" / "events_statements.csv", parse_dates=["meeting_date", "avail_date"]
    )
    grid = pd.read_csv(ROOT / "output" / "validation_grid.csv")
    fig_timeseries(scores)
    fig_announcement_scatter(events)
    fig_replication(grid)
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
