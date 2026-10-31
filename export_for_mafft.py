"""
Prepare one FASTA file for MAFFT: one mitogenome per species, all rotated to a common origin.

Why rotation: mitogenomes are circular and the records start at different positions
(Felidae in the control region, Canidae at tRNA-Phe, some Vulpes at COX1), and an
alignment of unrotated records is poor. All records are rotated so that they start
where a common anchor sequence (500 nt by default) begins; the strand is chosen by
the better match to the anchor.

Which record per species: one whose header contains the species name, then the smallest
share of characters other than A, C, G, T, then the longer record. A record is used for
one species only.

Examples (run in the folder that holds the FASTA folders):
  python export_for_mafft.py felidae_fastas felidae_for_mafft \
        --anchor-file felidae_fastas/Pardofelis_marmorata_complete.fasta \
        --anchor-accession MW257207.1 --anchor-start 0 --outgroup Crocuta_crocuta_complete.fasta
  # without L. adusta (partial record with 25 % ambiguous characters), outgroup added:
  python export_for_mafft.py canidae_fastas canidae28_for_mafft \
        --anchor-file canidae_fastas/Canis_familiaris_complete.fasta \
        --anchor-accession NC_002008.4 --anchor-start 0 \
        --drop Lupulella_adusta --outgroup Ursus_arctos_complete.fasta

Anchors: the anchor only fixes where the circular sequences are cut; it does not enter the
tree. It is a published record that starts at the beginning of the tRNA-Phe gene (GTTAATGTAGCTTAA...):
the marbled cat record MW257207.1 for Felidae and the dog RefSeq record NC_002008.4 for Canidae
(the first 500 nt are used). Any other record that starts there will do. Many Felidae records start
in the control region (for example the RefSeq record NC_028303.1 of the marbled cat), and cannot be
used with --anchor-start 0.

Output: <prefix>.fasta (species names as headers) and <prefix>_records.tsv
(accession, length, share of non-ACGT, rotation offset, strand, k-mer votes and the identity
of the first 120 nt with the anchor, for checking).
Then, for example:
  mafft --auto --thread 4 <prefix>.fasta > <prefix>_aln.fasta
  iqtree2 -s <prefix>_aln.fasta -m MFP -B 1000 -T AUTO -o Crocuta_crocuta
"""
import argparse
import os
import sys
from collections import Counter

K = 12
COMP = str.maketrans("ACGTRYMKWSBDHVN", "TGCAYRKMWSVHDBN")


def read_fasta(path):
    recs, name, seq = [], None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    recs.append((name, "".join(seq).upper()))
                name, seq = line[1:], []
            else:
                seq.append(line.strip())
    if name is not None:
        recs.append((name, "".join(seq).upper()))
    return recs


def revcomp(s):
    return s.translate(COMP)[::-1]


def rotation(seq, anchor_kmers):
    """Best start offset for a circular sequence and the strand; votes of the best and second best offset."""
    best = None
    for strand, s in (("+", seq), ("-", revcomp(seq))):
        L = len(s)
        d = s + s
        votes = Counter()
        for p in range(L + K):
            km = d[p:p + K]
            q = anchor_kmers.get(km)
            if q is not None:
                votes[(p - q) % L] += 1
        if votes:
            # merge offsets that differ by a few nt (insertions and deletions between species)
            merged = Counter()
            for off, v in votes.items():
                for o2 in range(off - 3, off + 4):
                    merged[o2 % L] += v
            off, v = merged.most_common(1)[0]
            # take the exact offset with the most votes inside the best window
            near = [(votes.get(o2 % L, 0), o2 % L) for o2 in range(off - 3, off + 4)]
            exact = max(near)[1]
            total = v
            second = [x for o, x in merged.most_common() if min((o - off) % L, (off - o) % L) > 50]
            res = (total, exact, strand, s, second[0] if second else 0)
            if best is None or res[0] > best[0]:
                best = res
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("prefix")
    ap.add_argument("--anchor-file", required=True, help="FASTA file with the record that defines the origin")
    ap.add_argument("--anchor-accession", help="use the record with this accession from the anchor file (default: the first record)")
    ap.add_argument("--anchor-start", type=int, required=True, help="0-based start of the anchor in that record")
    ap.add_argument("--anchor-len", type=int, default=500)
    ap.add_argument("--drop", nargs="*", default=[], help="species (file name stems) to leave out")
    ap.add_argument("--outgroup", nargs="*", default=[], help="FASTA files of outgroup species, added to the output")
    args = ap.parse_args()

    anchor_recs = read_fasta(args.anchor_file)
    if args.anchor_accession:
        anchor_recs = [r for r in anchor_recs if r[0].split()[0] == args.anchor_accession]
        if not anchor_recs:
            sys.exit("accession %s not found in %s" % (args.anchor_accession, args.anchor_file))
    anchor_rec = anchor_recs[0][1]
    anchor = anchor_rec[args.anchor_start:args.anchor_start + args.anchor_len]
    count = Counter(anchor[q:q + K] for q in range(len(anchor) - K + 1))
    anchor_kmers = {anchor[q:q + K]: q for q in range(len(anchor) - K + 1)
                    if count[anchor[q:q + K]] == 1 and set(anchor[q:q + K]) <= set("ACGT")}  # unique k-mers only

    files = [os.path.join(args.folder, f) for f in sorted(os.listdir(args.folder))
             if f.endswith((".fasta", ".fa", ".fas"))]
    files += list(args.outgroup)
    out, table, used = [], [], {}

    def nonacgt(seq):
        return sum(1 for c in seq if c not in "ACGT") / len(seq)

    def start_identity(rotated):
        """Identity of the first 120 nt with the anchor, allowing a shift of up to 8 nt (indels at the start)."""
        best = 0.0
        for sh in range(-8, 9):
            a = rotated[max(sh, 0):max(sh, 0) + 120]
            b = anchor[max(-sh, 0):max(-sh, 0) + 120]
            n = min(len(a), len(b))
            if n:
                best = max(best, sum(1 for x, y in zip(a, b) if x == y) / n)
        return round(best, 2)

    for path in files:
        species = os.path.basename(path).rsplit(".", 1)[0].replace("_complete", "")
        if species in args.drop:
            continue
        recs = read_fasta(path)
        binomial = species.replace("_", " ").lower()
        # header with the species name first, then the fewest non-ACGT characters, then the longest record
        ranked = sorted(recs, key=lambda r: (binomial not in r[0].lower(), nonacgt(r[1]), -len(r[1])))
        pick = next((r for r in ranked if r[0].split()[0] not in used), ranked[0])
        name, seq = pick
        acc = name.split()[0]
        if acc in used:
            print("WARNING: %s and %s end up with the same record %s" % (used[acc], species, acc))
        used.setdefault(acc, species)
        found = rotation(seq, anchor_kmers)
        if found is None:
            sys.exit("no 12-mer of %s matches the anchor; is the anchor-start right (the anchor must begin at the "
                     "wanted origin, and must not lie in a poorly conserved region)?" % species)
        votes, off, strand, s, second = found
        rotated = s[off:] + s[:off]
        out.append((species, rotated))
        table.append((species, acc, len(seq), round(100 * nonacgt(seq), 2), off, strand, votes, second,
                      start_identity(rotated)))

    with open(args.prefix + ".fasta", "w") as f:
        for species, seq in out:
            f.write(">%s\n" % species)
            for i in range(0, len(seq), 70):
                f.write(seq[i:i + 70] + "\n")
    with open(args.prefix + "_records.tsv", "w") as f:
        f.write("species\taccession\tlength\tpct_non_ACGT\trotation_offset\tstrand\tvotes\tvotes_second_best_offset\tstart_identity\n")
        for row in table:
            f.write("\t".join(str(x) for x in row) + "\n")

    weak = [r for r in table if r[6] < 30 or r[7] * 3 > r[6] or r[8] < 0.7]
    print("%d species written to %s.fasta" % (len(out), args.prefix))
    print("records with a weak or ambiguous rotation (check them):", [r[0] for r in weak] or "none")
    print("reverse-complemented:", [r[0] for r in table if r[5] == "-"] or "none")


if __name__ == "__main__":
    main()
