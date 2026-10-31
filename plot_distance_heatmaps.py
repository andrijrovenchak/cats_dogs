"""
plot_distance_heatmaps.py

Heatmaps of the species-by-species distance matrices written by
build_population_*_matrix (diagonal = largest distance between records of the
same species, off-diagonal = mean distance between records of two species).

Figures
  1. <family>_heatmaps.png   : the three matrices side by side, same species order
                               (order of the reference tree), genus blocks marked.
  2. <family>_<kind>_heatmap.png : one matrix with every species labelled (supplement).
  3. <family>_gap.png        : per species, the largest intra-species distance (filled)
                               against the smallest distance to another species (open).

Usage (in the folder with the csv files):
    python plot_distance_heatmaps.py                     # both families, suffix ''
    python plot_distance_heatmaps.py --suffix -28        # files like canidae_k2p_matrix-28.csv
"""
import sys
import argparse
import re

import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
import numpy as np
import pandas as pd

from mtdna import CANIDAE_REFERENCE, FELIDAE_REFERENCE, tuple_to_newick

KINDS = [("jaccard", "Jaccard (13-mers)"),
         ("k2p", "K2P (pairwise alignment)"),
         ("euclidean", "Euclidean (word presence)")]
REFS = {"felidae": FELIDAE_REFERENCE, "canidae": CANIDAE_REFERENCE}


def sampled_counts(folder, max_seqs=3):
    """Number of records per species that went into the matrices: min(records in file, max_seqs).
    A species with fewer than two has no within-species distance (the diagonal is 0 by construction)."""
    import os
    out = {}
    for fn in os.listdir(folder):
        if fn.endswith((".fasta", ".fa", ".fas")):
            n = 0
            with open(os.path.join(folder, fn)) as fh:
                for line in fh:
                    n += line.startswith(">")
            out[fn.replace("_complete.fasta", "").replace(".fasta", "")] = min(n, max_seqs)
    return out


def load_matrix(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.str.replace("_complete.fasta", "", regex=False)
    df.columns = df.columns.str.replace("_complete.fasta", "", regex=False)
    return df


def reference_order(ref, labels):
    """Species in the order of the leaves of the reference tree, restricted to `labels`."""
    names = re.findall(r"[A-Za-z]+_[a-z]+", tuple_to_newick(ref))
    seen, order = set(), []
    for n in names:
        if n in labels and n not in seen:
            seen.add(n)
            order.append(n)
    missing = [l for l in labels if l not in seen]
    return order + missing          # species not in the reference go last


def genus_blocks(order):
    blocks, start = [], 0
    for i in range(1, len(order) + 1):
        if i == len(order) or order[i].split("_")[0] != order[start].split("_")[0]:
            blocks.append((order[start].split("_")[0], start, i))
            start = i
    return blocks


def draw_heatmap(ax, df, title, labels=True, cmap="YlOrRd", pct=95):
    """Robust colour scale: the top of the scale is the `pct` percentile of the
    off-diagonal values; larger values are drawn in the darkest colour."""
    m = df.values.astype(float)
    off = m[~np.eye(len(m), dtype=bool)]
    vmax = np.nanpercentile(off, pct)
    n_clipped = int(np.sum(off > vmax))
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    order = list(df.index)
    for g, a, b in genus_blocks(order):
        if a > 0:
            ax.axhline(a - 0.5, color="0.45", lw=0.3)
            ax.axvline(a - 0.5, color="0.45", lw=0.3)
        if labels or b - a > 1:
            ax.text(-0.02, (a + b - 1) / 2, g, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=8 if labels else 9, fontweight="bold")
    n = len(order)
    ax.plot(range(n), range(n), "s", ms=1.2, mfc="none", mec="0.15", mew=0.3)   # diagonal = intra
    if labels:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ep = [o.split("_", 1)[1] for o in order]
        ax.set_xticklabels(ep, rotation=90, fontsize=5, fontstyle="italic")
        ax.set_yticklabels(ep, fontsize=5, fontstyle="italic")
        ax.tick_params(length=1.5, pad=1)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title, fontsize=12)
    return im, vmax, n_clipped


def panels(family, suffix, fasta_dir=None):
    mats = {}
    for kind, _ in KINDS:
        try:
            mats[kind] = load_matrix(f"{family}_{kind}_matrix{suffix}.csv")
        except FileNotFoundError:
            print(f"missing {family}_{kind}_matrix{suffix}.csv")
    if not mats:
        return
    labels = list(next(iter(mats.values())).index)
    order = reference_order(REFS[family], labels)
    fig, axes = plt.subplots(1, len(mats), figsize=(7.2, 2.7), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (kind, title) in zip(axes, [k for k in KINDS if k[0] in mats]):
        df = mats[kind].loc[order, order]
        im, vmax, nclip = draw_heatmap(ax, df, title, labels=False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, extend="max")
        cb.ax.tick_params(labelsize=6)
        print(f"{family} {kind}: colour scale 0-{vmax:.3f}, {nclip} cells above the scale")
        # single, fully labelled version for the supplement
        f2, a2 = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
        im2, _, _ = draw_heatmap(a2, df, f"{family.capitalize()}: {title}", labels=True)
        f2.colorbar(im2, ax=a2, fraction=0.04, pad=0.02, extend="max").ax.tick_params(labelsize=6)
        f2.savefig(f"{family}_{kind}_heatmap{suffix}.png", dpi=300)
        plt.close(f2)
    fig.savefig(f"{family}_heatmaps{suffix}.png", dpi=300)
    plt.close(fig)
    counts = sampled_counts(fasta_dir or f"{family}_dedup") if (fasta_dir or __import__("os").path.isdir(f"{family}_dedup")) else {}
    gap_plot(family, suffix, mats, order, counts)


FULL = {"Chrysocyon", "Cuon", "Lycalopex", "Lycaon"}     # genera written in full
ABBR = {"Lupulella": "Lu."}                              # special abbreviations

def short_label(name):
    g, s = name.split("_", 1)
    if g in FULL:
        return f"{g} {s}"
    return f"{ABBR.get(g, g[0] + '.')} {s}"


def draw_heatmap1(ax, df, title, cmap="YlOrRd", pct=95, fs=8, species_fs=6):
    """Heatmap with genus names on the left and species (epithets) on the bottom and the right.
    Robust colour scale: the top of the scale is the `pct` percentile of the off-diagonal
    values; larger values are drawn in the darkest colour."""
    m = df.values.astype(float)
    off = m[~np.eye(len(m), dtype=bool)]
    vmax = np.nanpercentile(off, pct)
    n_clipped = int(np.sum(off > vmax))
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    order = list(df.index)
    n = len(order)
    #ep = [f"{o[0]}. {o.split('_', 1)[1]}" for o in order]     # e.g. "P. tigris"
    ep = [short_label(o) for o in order]

    # genus names: left of the axis, 4 pt away
    tr = offset_copy(ax.get_yaxis_transform(), fig=ax.figure, x=-4, y=0, units="points")
    for g, a, b in genus_blocks(order):
        if a > 0:
            ax.axhline(a - 0.5, color="0.45", lw=0.3)
            ax.axvline(a - 0.5, color="0.45", lw=0.3)
        if b - a > 1:                       # genera with a single species are not named
            ax.text(0, (a + b - 1) / 2, g, transform=tr, ha="right", va="center",
                    fontsize=fs, fontweight="bold")

    ax.plot(range(n), range(n), "s", ms=1.2, mfc="none", mec="0.15", mew=0.3)   # diagonal = intra

    # species: bottom and right (the matrix is symmetric)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ep, rotation=90, fontsize=species_fs, fontstyle="italic")
    ax.set_yticklabels(ep, fontsize=species_fs, fontstyle="italic")
    ax.yaxis.tick_right()
    ax.tick_params(length=1.5, pad=1)
    ax.set_title(title, fontsize=fs + 1)
    return im, vmax, n_clipped


def panel1(family, kind, suffix, fasta_dir=None):
    title = dict(KINDS)[kind]
    try:
        df = load_matrix(f"{family}_{kind}_matrix{suffix}.csv")
    except FileNotFoundError:
        print(f"missing {family}_{kind}_matrix{suffix}.csv")
        return
    order = reference_order(REFS[family], list(df.index))
    df = df.loc[order, order]

    fig, ax = plt.subplots(figsize=(5.9, 6.2), constrained_layout=True)
    im, vmax, nclip = draw_heatmap1(ax, df, title, fs=8, species_fs=6)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", location="bottom",
                    fraction=0.04, pad=0.02, shrink=0.5, extend="max")
    cb.ax.tick_params(labelsize=8)

    print(f"{family} {kind}: colour scale 0-{vmax:.3f}, {nclip} cells above the scale")
    fig.savefig(f"{family}_{kind}_panel{suffix}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.show()
    plt.close(fig)

    # fully labelled version for the supplement
    f2, a2 = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    im2, _, _ = draw_heatmap(a2, df, f"{family.capitalize()}: {title}", labels=True)
    f2.colorbar(im2, ax=a2, fraction=0.04, pad=0.02, extend="max").ax.tick_params(labelsize=6)
    f2.savefig(f"{family}_{kind}_heatmap{suffix}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(f2)


def gap_plot(family, suffix, mats, order, counts):
    """Per species: largest intra-species distance (diagonal) against the smallest
    distance to any other species. A species is flagged when intra >= nearest inter."""
    fig, axes = plt.subplots(len(mats), 1, figsize=(7.2, 1.9 * len(mats)), sharex=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(order))
    for ax, (kind, title) in zip(axes, [k for k in KINDS if k[0] in mats]):
        df = mats[kind].loc[order, order]
        m = df.values.astype(float)
        intra = np.diag(m).copy()
        inter = np.where(np.eye(len(m), dtype=bool), np.inf, m).min(axis=1)
        single = np.array([counts.get(o, 2) < 2 for o in order])     # no within-species value
        bad = (intra >= inter) & ~single
        ax.vlines(x[~single], np.minimum(intra, inter)[~single], np.maximum(intra, inter)[~single], color="0.75", lw=0.8)
        ax.plot(x, inter, "o", mfc="none", mec="C0", ms=3.5, label="nearest other species")
        ax.vlines(x[single], 0, 0, color="none")
        ax.plot(x[~single], intra[~single], "o", color="C3", ms=3.0, label="largest within species")
        ax.plot(x[bad], intra[bad], "o", mfc="none", mec="k", ms=7, mew=0.7, label="within >= between")
        ax.set_ylabel(title.split(" (")[0], fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_yscale("symlog", linthresh=max(1e-3, float(np.nanmin(inter[inter > 0])) / 2))
        print(f"{family} {kind}: {int(bad.sum())} of {len(order)} species have within >= between:",
              [order[i].replace('_', ' ') for i in np.where(bad)[0]])
    axes[0].legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.55), frameon=False)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([o.replace("_", " ") + ("*" if counts.get(o, 2) < 2 else "") for o in order],
                             rotation=90, fontsize=5, fontstyle="italic")
    fig.text(0.995, 0.005, "* one record: no within-species distance", ha="right", va="bottom", fontsize=6)
    fig.savefig(f"{family}_gap{suffix}.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    # The two panels of the paper. 
    # Before running, rename the Canidae matrices so that the suffix gives the species count:
    #   canidae_jaccard_matrix-k13.csv -> canidae_jaccard_matrix-k13-28.csv
    #   (the same for the other Canidae matrices: canidae_k2p_matrix.csv -> canidae_k2p_matrix-28.csv, etc.)
    if len(sys.argv) == 1:
        panel1('felidae', 'k2p', '')
        panel1('canidae', 'jaccard', '-k13-28')
    else:
        # General use: all heatmaps of the chosen families (see the README)
        ap = argparse.ArgumentParser()
        ap.add_argument("--suffix", default="", help="e.g. -28 for canidae_k2p_matrix-28.csv")
        ap.add_argument("--families", nargs="+", default=["felidae", "canidae"])
        ap.add_argument("--fasta-dir", default=None, help="folder with the species FASTA files (default <family>_dedup)")
        args = ap.parse_args()
        for fam in args.families:
            panels(fam, args.suffix, args.fasta_dir)
