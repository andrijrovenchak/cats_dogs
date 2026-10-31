"""
Tanglegram: a UPGMA tree (left) against the reference topology (right), with a line
joining the two positions of each species.

Only matplotlib is needed (Newick files are read by a small parser below).

Branch colours show what the Robinson-Foulds distance counts:
  blue = clade (set of species) found in both trees,
  red  = clade found only in this tree.

Usage examples
--------------
# lowest-RF tree among the six distances, Felidae (44 species)
python tanglegram.py felidae

# Canidae, 29 species (or 28 without L. adusta)
python tanglegram.py canidae 29
python tanglegram.py canidae 28

# a chosen distance instead of the lowest-RF one
python tanglegram.py felidae --method jaccard

# explicit files
python tanglegram.py --tree felidae_tfidf_tree.nwk --reference reference_trees.nwk \
       --ref-key "Felidae reference tree, pruned" --out my_tanglegram

File names are expected as  {family}{n if canidae}_{method}_tree.nwk  (the rule used
for the supplement), e.g. felidae_tfidf_tree.nwk, canidae29_jaccard_tree.nwk.
"""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["jaccard", "mash", "k2p", "euclidean", "tfidf", "cosine-species", "euclidean-species"]

UPDATED_DISTANCE_NAMES = {
    "jaccard": "Jaccard (13-mers)",
    "mash": "Mash",
    "k2p": "K2P (pairwise alignment)",
    "euclidean": "Euclidean (genomes)",
    "tfidf": "TF-IDF cosine (species)",
    "cosine-species": "Cosine (species)",
    "euclidean-species": "Euclidean (species)",
}

# text that identifies the reference tree inside reference_trees.nwk (header lines start with '#')
REF_KEY = {
    ("felidae", None): "Felidae reference tree, pruned",
    ("canidae", 29): "Canidae reference tree, pruned to the species of the collection",
    ("canidae", 28): "without Lupulella_adusta",
}


# ----------------------------------------------------------------------------- Newick
class Node:
    __slots__ = ("name", "children", "y", "x", "leaves")

    def __init__(self, name=None):
        self.name = name
        self.children = []
        self.y = 0.0
        self.x = 0.0
        self.leaves = frozenset()


def parse_newick(text):
    """Parse one Newick string (whitespace and branch lengths are ignored)."""
    text = re.sub(r"\s+", "", text).rstrip(";")
    pos = 0

    def read_label():
        nonlocal pos
        m = re.match(r"[^,():;]*", text[pos:])
        pos += m.end()
        label = m.group(0)
        if pos < len(text) and text[pos] == ":":      # branch length
            m = re.match(r":[^,():;]*", text[pos:])
            pos += m.end()
        return label

    def read_node():
        nonlocal pos
        node = Node()
        if text[pos] == "(":
            pos += 1
            node.children.append(read_node())
            while text[pos] == ",":
                pos += 1
                node.children.append(read_node())
            assert text[pos] == ")", "unbalanced parentheses"
            pos += 1
            read_label()                              # internal label / support, not used
            if len(node.children) == 1:               # extra pair of parentheses
                return node.children[0]
        else:
            node.name = read_label()
        return node

    root = read_node()
    assert pos == len(text), "text left after the tree: %r" % text[pos:pos + 20]
    return root


def read_tree_file(path):
    """File holding one tree (lines starting with '#' are skipped)."""
    with open(path) as f:
        lines = [l for l in f if not l.lstrip().startswith("#")]
    return parse_newick("".join(lines))


def read_reference(path, key):
    """Tree from reference_trees.nwk whose '#' header line contains `key`."""
    blocks, header, buf = [], None, []
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                if header is not None and buf:
                    blocks.append((header, "".join(buf)))
                header, buf = line, []
            else:
                buf.append(line)
    if header is not None and buf:
        blocks.append((header, "".join(buf)))
    hits = [(h, t) for h, t in blocks if key in h and t.strip().startswith("(")]
    if len(hits) != 1:
        sys.exit("reference: %d trees match %r; headers: %s"
                 % (len(hits), key, [h.strip() for h, _ in blocks]))
    return parse_newick(hits[0][1])


# ----------------------------------------------------------------------------- tree tools
def leaf_names(node):
    return [node.name] if not node.children else [n for c in node.children for n in leaf_names(c)]


def prune(node, keep):
    """Remove leaves not in `keep`; unary nodes are collapsed."""
    if not node.children:
        return node if node.name in keep else None
    kids = [k for k in (prune(c, keep) for c in node.children) if k is not None]
    if not kids:
        return None
    if len(kids) == 1:
        return kids[0]
    node.children = kids
    return node


def set_leaves(node):
    node.leaves = frozenset([node.name]) if not node.children else \
        frozenset().union(*[set_leaves(c) for c in node.children])
    return node.leaves


def clades(node):
    """Non-trivial clades (2 <= size < n) of a rooted tree, as a set of frozensets."""
    out = set()
    n = len(node.leaves)

    def walk(v):
        if v.children and 1 < len(v.leaves) < n:
            out.add(v.leaves)
        for c in v.children:
            walk(c)
    walk(node)
    return out


def rf(a, b):
    """Rooted Robinson-Foulds distance and its maximum, 2(n-2)."""
    ca, cb = clades(a), clades(b)
    return len(ca ^ cb), 2 * (len(a.leaves) - 2)


def rotate_like(node, position):
    """Order the children of every node by the mean position of their leaves in the other tree."""
    for c in node.children:
        rotate_like(c, position)
    node.children.sort(key=lambda c: sum(position[l] for l in c.leaves) / len(c.leaves))


def layout(root):
    """y: leaves 0,1,2,... top to bottom, internal nodes at the mean of their children;
    x: cladogram, tips at 0 and every node one unit above its highest child."""
    counter = [0]

    def walk(v):
        if not v.children:
            v.y = counter[0]
            counter[0] += 1
            v.x = 0.0
        else:
            for c in v.children:
                walk(c)
            v.y = (min(c.y for c in v.children) + max(c.y for c in v.children)) / 2.0
            v.x = max(c.x for c in v.children) + 1.0
    walk(root)
    return root.x


# ----------------------------------------------------------------------------- drawing
BLUE, RED, GREY = "#1f5fa8", "#c0392b", "#9a9a9a"


def draw_tree(ax, root, other_clades, x0, width, direction):
    """direction = +1: root on the left, tips on the right (left tree);
       direction = -1: mirrored (right tree).  x0 is the tip position."""
    height = root.x
    scale = width / height

    def px(v):
        return x0 - direction * v.x * scale

    def walk(v, parent):
        if parent is not None:
            if v.children:
                col = BLUE if v.leaves in other_clades else RED
            else:
                col = GREY
            ax.plot([px(parent), px(v)], [v.y, v.y], color=col, lw=1.2, solid_capstyle="butt")
        if v.children:
            ys = [c.y for c in v.children]
            ax.plot([px(v), px(v)], [min(ys), max(ys)], color="#444444", lw=0.8)
            for c in v.children:
                walk(c, v)
    walk(root, None)


def tanglegram(left, right, left_title, right_title, out, label_size=6.5):
    names = leaf_names(left)
    n = len(names)
    # the right tree is rotated to follow the left one (fewer crossing lines)
    set_leaves(left)
    set_leaves(right)
    layout(left)
    pos_left = {}

    def collect(v):
        if not v.children:
            pos_left[v.name] = v.y
        for c in v.children:
            collect(c)
    collect(left)
    rotate_like(right, pos_left)
    layout(left)
    layout(right)

    cl_left, cl_right = clades(left), clades(right)
    rf_val, rf_max = rf(left, right)

    fig_h = max(6.0, 0.19 * n + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    gap = 1.0                                  # central strip: names, then the lines to the right tree
    width = 0.8
    draw_tree(ax, left, cl_right, x0=width, width=width, direction=+1)
    draw_tree(ax, right, cl_left, x0=width + gap, width=width, direction=-1)

    y_right = {}

    def collect_r(v):
        if not v.children:
            y_right[v.name] = v.y
        for c in v.children:
            collect_r(c)
    collect_r(right)

    xa, xb = width, width + gap                # tips of the left and right tree
    x_text = xa + 0.06                         # names start here, in the rows of the left tree
    x_line = xa + 0.44                         # lines to the right tree start here
    for name in names:
        ya, yb = pos_left[name], y_right[name]
        ax.text(x_text, ya, name.replace("_", " "), fontsize=label_size, style="italic",
                ha="left", va="center", zorder=3)
        ax.plot([x_line, xb], [ya, yb], color=GREY, lw=0.6, alpha=0.9, zorder=0)

    ax.set_xlim(-0.05, 2 * width + gap + 0.05)
    ax.set_ylim(n - 0.3, -0.9)                 # first species at the top
    ax.axis("off")
    ax.text(width / 2, -1.3, left_title, ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text(width + gap + width / 2, -1.3, right_title, ha="center", va="bottom",
            fontsize=10, weight="bold")
    ax.plot([], [], color=BLUE, lw=2, label="clade in both trees")
    ax.plot([], [], color=RED, lw=2, label="clade in this tree only")
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=8,
              bbox_to_anchor=(0.5, 0.0))
    ax.text(width + gap / 2, -1.3,
            "RF distance %d of maximum %d (%.0f%%)\n%d of %d reference clades found"
            % (rf_val, rf_max, 100.0 * rf_val / rf_max, len(cl_left & cl_right), len(cl_right)),
            ha="center", va="bottom", fontsize=9)
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return rf_val, rf_max


# ----------------------------------------------------------------------------- driver
def find_tree(folder, family, n, method):
    name = "%s%s_%s_tree.nwk" % (family, n if family == "canidae" else "", method)
    path = os.path.join(folder, name)
    return path if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family", nargs="?", choices=["felidae", "canidae"])
    ap.add_argument("n", nargs="?", type=int, help="canidae only: 29 or 28 species")
    ap.add_argument("--folder", default=".", help="folder with the *_tree.nwk files")
    ap.add_argument("--reference", default="reference_trees.nwk")
    ap.add_argument("--method", choices=METHODS, help="use this distance instead of the lowest-RF one")
    ap.add_argument("--tree", help="explicit UPGMA tree file")
    ap.add_argument("--ref-key", help="text of the '#' header of the reference tree")
    ap.add_argument("--out", help="output name without extension")
    args = ap.parse_args()

    if args.tree:
        key = args.ref_key or sys.exit("--ref-key is needed with --tree")
        cand = [(os.path.basename(args.tree), args.tree)]
        tag = os.path.splitext(os.path.basename(args.tree))[0]
    else:
        if not args.family:
            sys.exit("give felidae or canidae (and 29 or 28), or --tree")
        n = args.n if args.family == "canidae" else None
        if args.family == "canidae" and n not in (28, 29):
            sys.exit("canidae needs 29 or 28")
        key = args.ref_key or REF_KEY[(args.family, n)]
        methods = [args.method] if args.method else METHODS
        cand = [(m, find_tree(args.folder, args.family, n, m)) for m in methods]
        cand = [(m, p) for m, p in cand if p]
        tag = "%s%s" % (args.family, n if n else "")
    if not cand:
        sys.exit("no tree files found")

    ref = read_reference(args.reference, key)
    ref_leaves = set(leaf_names(ref))

    scored = []
    for m, p in cand:
        t = read_tree_file(p)
        if set(leaf_names(t)) != ref_leaves:
            print("WARNING %s: leaf sets differ from the reference (only in tree: %s; only in reference: %s)"
                  % (p, sorted(set(leaf_names(t)) - ref_leaves), sorted(ref_leaves - set(leaf_names(t)))))
            t = prune(t, ref_leaves)
        set_leaves(t)
        set_leaves(ref)
        scored.append((rf(t, ref)[0], m, p, t))
    scored.sort(key=lambda s: (s[0], METHODS.index(s[1]) if s[1] in METHODS else 99))

    print("RF to the reference (max %d):" % (2 * (len(ref_leaves) - 2)))
    for v, m, p, _ in scored:
        print("  %-16s %3d   %s" % (m, v, p))
    best = scored[0]
    ties = [s[1] for s in scored if s[0] == best[0]]
    if len(ties) > 1 and not args.method:
        print("Tie for the lowest RF: %s. Using %s; pick another with --method." % (ties, best[1]))

    out = args.out or "tanglegram_%s_%s" % (tag, best[1])
    # fresh copy of the reference for drawing (the right tree is rotated)
    ref = read_reference(args.reference, key)
    set_leaves(ref)
    rf_val, rf_max = tanglegram(best[3], ref,
                            "UPGMA tree\n%s" % UPDATED_DISTANCE_NAMES.get(best[1], best[1]),
                            "Reference topology\n(pruned to %d species)" % len(ref_leaves),
                            out)
    print("written %s.pdf and %s.png (RF %d/%d)" % (out, out, rf_val, rf_max))


if __name__ == "__main__":
    main()
