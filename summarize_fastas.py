#!/usr/bin/env python3
"""
summarize_fastas.py -- stage 3 of the Cats&Dogs pipeline: summary and tables.

Reads the cleaned FASTA folders (one multi-FASTA file per species, as written
by collect_fastas.py + remove_shared_records.py) and writes:

  records.csv              one row per record (composition, ambiguity codes, tokens, chunks, sampled?)
  species.csv              one row per species
  accessions_supplement.csv  accession list for the supplementary table (family, species, accession, label, length, sampled)
  family_summary.tex       booktabs table for the main text (one row per family)
  species_table.tex        booktabs table for a supplement (one row per species)
  checks.txt               data checks: shared accessions, identical sequences across species,
                           strand flags, odd lengths, ambiguity codes, missing species

The sampling and chunk counts repeat the logic of mtdna.get_tokenized_documents
(random.seed(seed); random.sample(records, max_seqs) for files with more records
than max_seqs; split at 'A'; windows of `words` tokens every `step` tokens, last
windows kept). Use --verify-mtdna DIR to compare the chunk counts with the real
function (DIR is the folder that contains mtdna.py).

Only the standard library is needed.

Example
-------
python summarize_fastas.py felidae_dedup canidae_dedup --out summary \
       --expected-felidae felidae_species.txt --expected-canidae canidae_species.txt
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import math
import os
import random
import re
import statistics
import sys

EXTS = (".fasta", ".fa", ".fas")
AMBIG_WARN_FRACTION = 0.01          # flag records with >= 1 % non-ACGT
G_EXPECTED = (0.10, 0.17)           # light-strand G fraction range for vertebrate mtDNA
LEN_EXPECTED = (15000, 19000)       # plausible mitogenome lengths
MOTIF = "GTTAATGTAGCTTA"            # tRNA-Phe-like motif, used only to describe where records start


# --------------------------------------------------------------------------- IO
def parse_fasta(path: str) -> list[tuple[str, str]]:
    records, header, buf = [], None, []
    with open(path, newline="") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(buf).upper()))
                header, buf = line[1:], []
            elif line.strip():
                buf.append(line.strip())
    if header is not None:
        records.append((header, "".join(buf).upper()))
    return records


def species_from_filename(fname: str) -> str:
    stem = os.path.splitext(fname)[0]
    return re.sub(r"_(complete|refseq|any)$", "", stem)


def list_fasta_files(folder: str) -> list[str]:
    return sorted(f for f in os.listdir(folder) if f.endswith(EXTS))   # same order as the analysis scripts


def pretty(species: str) -> str:
    return species.replace("_", " ")


# ------------------------------------------------------------------ calculations
def tokenise_counts(seq: str, step: int, words: int, short: int) -> tuple[int, int, int]:
    m = seq.count("A") + 1                              # number of tokens after splitting at A
    starts = range(0, m, step)
    n_chunks = len(starts)
    n_short = sum(1 for i in starts if min(words, m - i) < short)
    return m, n_chunks, n_short


def sample_indices(n_records: int, max_seqs: int, seed: int) -> list[int]:
    if n_records > max_seqs:
        random.seed(seed)
        return random.sample(range(n_records), max_seqs)
    return list(range(n_records))


def analyse_family(family: str, folder: str, args) -> list[dict]:
    rows = []
    for fname in list_fasta_files(folder):
        species = species_from_filename(fname)
        recs = parse_fasta(os.path.join(folder, fname))
        chosen = set(sample_indices(len(recs), args.max_seqs, args.seed))
        for idx, (header, seq) in enumerate(recs):
            n = len(seq)
            comp = collections.Counter(seq)
            ambig = {k: v for k, v in comp.items() if k not in "ACGT"}
            m, n_chunks, n_short = tokenise_counts(seq, args.step, args.words, args.short)
            motif_pos = seq.find(MOTIF)
            parts = header.split(None, 1)
            rows.append(dict(
                family=family, species=species, file=fname, record_index=idx,
                accession=parts[0], label=parts[1] if len(parts) > 1 else "",
                length=n,
                frac_A=comp["A"] / n if n else 0, frac_C=comp["C"] / n if n else 0,
                frac_G=comp["G"] / n if n else 0, frac_T=comp["T"] / n if n else 0,
                n_ambiguous=sum(ambig.values()),
                ambiguous_codes="".join(sorted(ambig)),
                tokens=m, chunks=n_chunks, short_chunks=n_short,
                motif_pos=motif_pos,
                sampled=int(idx in chosen),
                _seq=seq,
            ))
    return rows


def species_rows(rows: list[dict]) -> list[dict]:
    by_sp = collections.OrderedDict()
    for r in rows:
        by_sp.setdefault((r["family"], r["species"]), []).append(r)
    out = []
    for (family, species), rs in by_sp.items():
        lens = [r["length"] for r in rs]
        gs = [r["frac_G"] for r in rs]
        samp = [r for r in rs if r["sampled"]]
        out.append(dict(
            family=family, species=species, records=len(rs),
            unique_sequences=len({r["_seq"] for r in rs}),
            len_min=min(lens), len_median=int(statistics.median(lens)), len_max=max(lens),
            G_min=min(gs), G_max=max(gs),
            records_with_ambiguity=sum(1 for r in rs if r["n_ambiguous"]),
            ambiguous_bases=sum(r["n_ambiguous"] for r in rs),
            sampled=len(samp), chunks=sum(r["chunks"] for r in samp),
            short_chunks=sum(r["short_chunks"] for r in samp),
        ))
    return out


# ------------------------------------------------------------------------ output
def write_csv(path: str, rows: list[dict], skip=("_seq",)) -> None:
    if not rows:
        return
    cols = [c for c in rows[0] if c not in skip]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fmt_int(x: int) -> str:
    return f"{x:,}".replace(",", "{,}")


def family_table(rows: list[dict], sp_rows: list[dict], args) -> str:
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Family & Species & Records & Sampled & Chunks & Length (bp) & Records with N/IUPAC \\",
             r"\midrule"]
    for fam in sorted({r["family"] for r in rows}):
        rs = [r for r in rows if r["family"] == fam]
        ss = [s for s in sp_rows if s["family"] == fam]
        samp = [r for r in rs if r["sampled"]]
        lens = [r["length"] for r in rs]
        lines.append(
            f"{fam} & {len(ss)} & {fmt_int(len(rs))} & {len(samp)} & {fmt_int(sum(r['chunks'] for r in samp))} & "
            f"{fmt_int(min(lens))}--{fmt_int(max(lens))} & {sum(1 for r in rs if r['n_ambiguous'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def species_table(sp_rows: list[dict]) -> str:
    lines = [r"\begin{longtable}{llrrrr}",
             r"\caption{Species included in the analysis.}\label{tab:species-supplement}\\",
             r"\toprule",
             r"Family & Species & Records & Distinct sequences & Length (bp) & Sampled \\",
             r"\midrule", r"\endfirsthead", r"\toprule",
             r"Family & Species & Records & Distinct sequences & Length (bp) & Sampled \\",
             r"\midrule", r"\endhead"]
    for s in sp_rows:
        length = (fmt_int(s["len_min"]) if s["len_min"] == s["len_max"]
                  else f"{fmt_int(s['len_min'])}--{fmt_int(s['len_max'])}")
        lines.append(f"{s['family']} & \\textit{{{pretty(s['species'])}}} & {fmt_int(s['records'])} & "
                     f"{fmt_int(s['unique_sequences'])} & {length} & {s['sampled']} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------ checks
def run_checks(rows: list[dict], sp_rows: list[dict], expected: dict[str, list[str]]) -> list[str]:
    out = []
    add = out.append

    # 1. shared accessions between species files
    by_acc = collections.defaultdict(set)
    for r in rows:
        by_acc[r["accession"]].add(r["species"])
    shared = {a: s for a, s in by_acc.items() if len(s) > 1}
    add(f"[1] Accessions present in more than one species file: {len(shared)}"
        + ("  (OK)" if not shared else "  <-- run remove_shared_records.py or fix the parent/child table"))
    pairs = collections.Counter(tuple(sorted(s)) for s in shared.values())
    for p, n in pairs.most_common(15):
        add(f"      {' / '.join(p)}: {n}")

    # 2. identical sequences across different species (different accessions)
    by_seq = collections.defaultdict(list)
    for r in rows:
        by_seq[r["_seq"]].append(r)
    cross = collections.Counter()
    for recs in by_seq.values():
        sp = sorted({r["species"] for r in recs})
        if len(sp) > 1:
            cross[tuple(sp)] += 1
    add(f"[2] Identical sequences shared by different species: {len(cross)} species group(s)")
    for sp, n in cross.most_common(15):
        add(f"      {' / '.join(sp)}: {n} distinct sequence(s)")

    # 3. identical sequences inside species
    dup_sp = [s for s in sp_rows if s["unique_sequences"] < s["records"]]
    add(f"[3] Species whose records contain repeated identical sequences: {len(dup_sp)} of {len(sp_rows)}")
    for s in sorted(dup_sp, key=lambda s: s["records"] - s["unique_sequences"], reverse=True)[:10]:
        add(f"      {s['species']}: {s['records']} records, {s['unique_sequences']} distinct sequences")

    # 4. strand
    odd_g = [r for r in rows if not (G_EXPECTED[0] <= r["frac_G"] <= G_EXPECTED[1])]
    add(f"[4] Records with G fraction outside {G_EXPECTED[0]:.2f}-{G_EXPECTED[1]:.2f} (possible reverse complement): {len(odd_g)}"
        + ("  (OK)" if not odd_g else ""))
    for r in odd_g[:15]:
        add(f"      {r['species']} {r['accession']} G={r['frac_G']:.3f}")

    # 5. lengths
    odd_len = [r for r in rows if not (LEN_EXPECTED[0] <= r["length"] <= LEN_EXPECTED[1])]
    add(f"[5] Records with length outside {LEN_EXPECTED[0]}-{LEN_EXPECTED[1]} bp: {len(odd_len)}"
        + ("  (OK)" if not odd_len else ""))
    for r in odd_len[:15]:
        add(f"      {r['species']} {r['accession']} {r['length']} bp")

    # 6. ambiguity codes
    amb = [r for r in rows if r["n_ambiguous"]]
    heavy = [r for r in amb if r["n_ambiguous"] / r["length"] >= AMBIG_WARN_FRACTION]
    add(f"[6] Records with non-ACGT characters: {len(amb)} of {len(rows)}; "
        f"with >= {AMBIG_WARN_FRACTION:.0%} non-ACGT: {len(heavy)}")
    sampled_amb = [r for r in amb if r["sampled"]]
    add(f"      among sampled records: {len(sampled_amb)} of {sum(r['sampled'] for r in rows)}")
    for r in sorted(heavy, key=lambda r: -r["n_ambiguous"])[:10]:
        add(f"      {r['species']} {r['accession']}: {r['n_ambiguous']} ({r['ambiguous_codes']})")

    # 7. short windows
    samp = [r for r in rows if r["sampled"]]
    add(f"[7] Chunks in sampled records: {sum(r['chunks'] for r in samp)}; "
        f"shorter than the --short threshold: {sum(r['short_chunks'] for r in samp)}")

    # 8. record origin (descriptive only)
    pos = [r["motif_pos"] for r in samp if r["motif_pos"] >= 0]
    by_fam = collections.defaultdict(list)
    for r in samp:
        if r["motif_pos"] >= 0:
            by_fam[r["family"]].append(r["motif_pos"])
    add(f"[8] tRNA-Phe-like motif found in {len(pos)} of {len(samp)} sampled records (start of the record = position 0):")
    for fam, ps in sorted(by_fam.items()):
        add(f"      {fam}: median position {int(statistics.median(ps))}, {sum(p < 60 for p in ps)} of {len(ps)} within the first 60 nt")

    # 9. one-record species
    single = [s["species"] for s in sp_rows if s["records"] == 1]
    add(f"[9] Species represented by a single record: {len(single)}" + (f" ({', '.join(single)})" if single else ""))

    # 10. expected species list
    for fam, names in expected.items():
        have = {s["species"] for s in sp_rows if s["family"] == fam}
        want = {n.strip().replace(" ", "_") for n in names if n.strip()}
        missing = sorted(want - have)
        extra = sorted(have - want)
        add(f"[10] {fam}: expected {len(want)}, found {len(have)}; missing: {len(missing)}; not in expected list: {len(extra)}")
        if missing:
            add("      missing: " + ", ".join(missing))
        if extra:
            add("      extra:   " + ", ".join(extra))
    return out


def verify_with_mtdna(folder_by_family: dict[str, str], rows: list[dict], mtdna_dir: str, args) -> list[str]:
    sys.path.insert(0, mtdna_dir)
    try:
        from mtdna import get_tokenized_documents          # type: ignore
    except Exception as exc:                               # noqa: BLE001
        return [f"[verify] could not import mtdna from {mtdna_dir}: {exc}"]
    bad = 0
    total = 0
    for fam, folder in folder_by_family.items():
        for fname in list_fasta_files(folder):
            docs = get_tokenized_documents(os.path.join(folder, fname), space_char="A",
                                           words_per_chunk=args.words, step_size=args.step,
                                           max_seqs_per_file=args.max_seqs)
            mine = sum(r["chunks"] for r in rows if r["file"] == fname and r["family"] == fam and r["sampled"])
            total += 1
            if len(docs) != mine:
                bad += 1
                print(f"  chunk count differs for {fname}: mtdna={len(docs)} here={mine}", file=sys.stderr)
    return [f"[verify] compared {total} files with mtdna.get_tokenized_documents: "
            + ("all chunk counts agree" if not bad else f"{bad} file(s) differ")]


# -------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("felidae_dir")
    ap.add_argument("canidae_dir")
    ap.add_argument("--out", default="summary")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seqs", type=int, default=3, help="records sampled per species file (as in the analysis scripts)")
    ap.add_argument("--words", type=int, default=150, help="window length in tokens")
    ap.add_argument("--step", type=int, default=75, help="window step in tokens")
    ap.add_argument("--short", type=int, default=100, help="windows with fewer tokens are counted as short")
    ap.add_argument("--expected-felidae", help="text file, one species per line, to detect missing species")
    ap.add_argument("--expected-canidae", help="text file, one species per line")
    ap.add_argument("--verify-mtdna", metavar="DIR", help="folder containing mtdna.py; compares chunk counts")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    folders = {"Felidae": args.felidae_dir, "Canidae": args.canidae_dir}
    rows: list[dict] = []
    for fam, folder in folders.items():
        rows += analyse_family(fam, folder, args)
    sp_rows = species_rows(rows)

    write_csv(os.path.join(args.out, "records.csv"), rows)
    write_csv(os.path.join(args.out, "species.csv"), sp_rows)
    acc_rows = [dict(family=r["family"], species=r["species"], accession=r["accession"],
                     label=r["label"], length=r["length"], sampled=r["sampled"]) for r in rows]
    write_csv(os.path.join(args.out, "accessions_supplement.csv"), acc_rows)
    with open(os.path.join(args.out, "family_summary.tex"), "w", encoding="utf-8") as fh:
        fh.write(family_table(rows, sp_rows, args))
    with open(os.path.join(args.out, "species_table.tex"), "w", encoding="utf-8") as fh:
        fh.write(species_table(sp_rows))

    expected = {}
    for fam, path in (("Felidae", args.expected_felidae), ("Canidae", args.expected_canidae)):
        if path:
            with open(path, encoding="utf-8") as fh:
                expected[fam] = [l for l in fh.read().splitlines()]

    report = [f"summarize_fastas.py   {datetime.date.today().isoformat()}   python {sys.version.split()[0]}",
              f"parameters: seed={args.seed} max_seqs={args.max_seqs} words={args.words} step={args.step} short<{args.short}",
              ""]
    for fam in folders:
        rs = [r for r in rows if r["family"] == fam]
        ss = [s for s in sp_rows if s["family"] == fam]
        samp = [r for r in rs if r["sampled"]]
        report.append(f"{fam}: {len(ss)} species, {len(rs)} records, {len(samp)} sampled, "
                      f"{sum(r['chunks'] for r in samp)} chunks")
    report.append("")
    report += run_checks(rows, sp_rows, expected)
    if args.verify_mtdna:
        report.append("")
        report += verify_with_mtdna(folders, rows, args.verify_mtdna, args)

    text = "\n".join(report) + "\n"
    with open(os.path.join(args.out, "checks.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"Files written to {args.out}/")


if __name__ == "__main__":
    main()
