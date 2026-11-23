"""Milestone 4 figures from the logged results (no simulation needed).

    python scripts/make_figures.py
Writes docs/figures/m4_success.png, m4_learning_curve.png, m4_outcome_map.png and prints the numbers used.
Colors: validated categorical slots 1/2 (blue = SFT only, orange = SFT + residual SAC) on surface #fcfcfb.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from newbody.evaluation.stats import wilson_interval

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs/figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#1a1a19", "#5f5e58", "#e4e3df"
C_SFT, C_RL = "#2a78d6", "#eb6834"
BIN1 = np.array([-0.05, -0.25])
DEMO_HALF = (0.055, 0.075)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def read(path):
    return list(csv.DictReader(open(ROOT / path)))


def rolling(x, w):
    x = np.asarray(x, float)
    return np.array([x[max(0, i - w + 1): i + 1].mean() for i in range(len(x))])


def success_figure(base, rl):
    def stats(rows, cond):
        s = [r for r in rows if cond(r)]
        k = sum(int(r["success"]) for r in s)
        return k, len(s)

    inside = lambda r: int(r["inside"]) == 1
    groups = [("All test seeds", lambda r: True), ("Inside demo region", inside),
              ("Outside demo region", lambda r: not inside(r))]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    y = np.arange(len(groups))[::-1]
    out = {}
    for j, (rows, name, color, dy) in enumerate(((base, "SFT only", C_SFT, 0.18), (rl, "SFT + residual SAC", C_RL, -0.18))):
        for yi, (g, cond) in zip(y, groups):
            k, n = stats(rows, cond)
            lo, hi = wilson_interval(k, n)
            p = k / n
            ax.plot([lo * 100, hi * 100], [yi + dy] * 2, color=color, lw=2, solid_capstyle="round")
            ax.plot(p * 100, yi + dy, "o", ms=8, color=color, mec=SURFACE, mew=2, label=name if yi == y[0] else None)
            ax.text(hi * 100 + 1.2, yi + dy, f"{p:.0%} ({k}/{n})", va="center", fontsize=9, color=INK)
            out[f"{name} | {g}"] = (k, n, round(lo, 3), round(hi, 3))
    ax.set_yticks(y, [g for g, _ in groups])
    ax.set_xlim(30, 112)
    ax.set_xticks(range(30, 101, 10))
    ax.set_xlabel("Success rate (%), dot = estimate, line = 95% Wilson interval")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    ax.set_title("Test success, 200 paired seeds (can anywhere in the source bin)", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "m4_success.png", dpi=150)
    return out


def learning_curve_figure():
    ep = read("outputs/rl_v1/episodes.csv")
    up = read("outputs/rl_v1/updates.csv")
    ev = read("outputs/rl_v1/evals.csv")
    sel = {249: 32, 349: 31, 399: 33, 449: 29, 499: 34, 549: 29, 599: 27}  # RL-val of saved checkpoints
    x = np.array([int(r["episode"]) for r in ep])
    fig, axs = plt.subplots(3, 1, figsize=(7.6, 7.4), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1, 1]))

    ax = axs[0]
    roll = 100 * rolling([int(r["success"]) for r in ep], 100)
    ax.plot(x[30:], roll[30:], color=C_RL, lw=2,
            label="training episodes (stochastic actor), 100-episode mean")
    ex = [int(r["episode"]) for r in ev]
    ax.plot(ex, [100 * int(r["success"]) / int(r["n"]) for r in ev], "o", ms=8, color=C_RL, mfc=SURFACE, mew=2,
            label="RL-val, deterministic actor (40 fixed seeds)")
    ax.plot([k + 1 for k in sel], [100 * v / 40 for v in sel.values()], "s", ms=6, color=C_RL, mfc=SURFACE, mew=1.5,
            label="RL-val of saved checkpoints (selection)")
    ax.axhline(100 * 29 / 40, color=C_SFT, lw=2, ls=(0, (4, 3)))
    ax.text(5, 100 * 29 / 40 + 1.5, "SFT only on RL-val (29/40)", color=INK2, fontsize=8)
    ax.set_ylabel("Success (%)")
    ax.set_ylim(20, 100)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.set_title("rl_v1: improvement until ~episode 500, then off-policy degradation", loc="left", fontsize=11)

    axs[1].plot(x, rolling([float(r["mean_abs_delta"]) for r in ep], 50), color=C_RL, lw=2)
    axs[1].set_ylabel("mean |δ|\n(50-ep mean)")
    ux = [int(r["episode"]) for r in up]
    axs[2].plot(ux, rolling([float(r["q_loss"]) for r in up], 20), color=C_RL, lw=2)
    axs[2].set_ylabel("critic loss\n(20-ep mean)")
    axs[2].set_xlabel("Training episode (1 episode ≈ 13–30 RL decisions of 10 env steps)")
    for a in axs:
        a.axvline(500, color=INK2, lw=1, ls=":")
        a.axvline(763, color=INK2, lw=1, ls=":")
    axs[0].text(505, 96, "selected\ncheckpoint", fontsize=8, color=INK2, va="top")
    axs[0].text(758, 96, "stopped", fontsize=8, color=INK2, va="top", ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "m4_learning_curve.png", dpi=150)


def outcome_map_figure(base, rl):
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 6.0), sharey=True)
    for ax, rows, name, color in ((axs[0], base, "SFT only", C_SFT), (axs[1], rl, "SFT + residual SAC", C_RL)):
        ok = [r for r in rows if int(r["success"])]
        bad = [r for r in rows if not int(r["success"])]
        ax.add_patch(Rectangle((-DEMO_HALF[0] * 100, -DEMO_HALF[1] * 100), 2 * DEMO_HALF[0] * 100,
                               2 * DEMO_HALF[1] * 100, fill=False, ec=INK2, lw=1.2, ls="--"))
        ax.text(DEMO_HALF[0] * 100 + 0.5, -DEMO_HALF[1] * 100, "demo\nregion", fontsize=8, color=INK2, va="bottom")
        ax.scatter([float(r["x"]) for r in ok], [float(r["y"]) for r in ok], s=34, color=color,
                   edgecolors=SURFACE, linewidths=1.2, label=f"success ({len(ok)})")
        ax.scatter([float(r["x"]) for r in bad], [float(r["y"]) for r in bad], s=34, marker="x", color=INK2,
                   linewidths=1.6, label=f"failure ({len(bad)})")
        ax.set_title(f"{name}: {len(ok)} success (dot), {len(bad)} failure (×)", loc="left", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("can start x rel. bin center (cm)")
        ax.set_xlim(-15, 15)
        ax.set_ylim(-19, 19)
    axs[0].set_ylabel("can start y rel. bin center (cm)")
    fig.suptitle("Test outcomes by can start position (same 200 seeds in both panels; +y points toward the target bin)",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    fig.savefig(FIG / "m4_outcome_map.png", dpi=150)


def main():
    b = {int(r["seed"]): r for r in read("results/eval/sft_v1_010000_testwide_k10/episodes.csv")}
    rl_rows = read("results/eval/rl_v1_ep499_test/episodes.csv")
    base, rl = [], []
    for r in rl_rows:
        s = int(r["seed"])
        x = (float(b[s]["can_x0"]) - BIN1[0]) * 100
        y = (float(b[s]["can_y0"]) - BIN1[1]) * 100
        inside = int(r["inside_demo_region"])
        base.append(dict(seed=s, success=b[s]["success"], inside=inside, x=x, y=y, mode=b[s]["failure_mode"]))
        rl.append(dict(seed=s, success=r["success"], inside=inside, x=x, y=y, mode=r["failure_mode"]))
    nums = success_figure(base, rl)
    learning_curve_figure()
    outcome_map_figure(base, rl)
    pairs = dict(both=sum(int(a["success"]) and int(c["success"]) for a, c in zip(base, rl)),
                 only_rl=sum((not int(a["success"])) and int(c["success"]) for a, c in zip(base, rl)),
                 only_sft=sum(int(a["success"]) and not int(c["success"]) for a, c in zip(base, rl)),
                 neither=sum((not int(a["success"])) and not int(c["success"]) for a, c in zip(base, rl)))
    print(json.dumps(dict(success=nums, paired=pairs), indent=2))


if __name__ == "__main__":
    main()
