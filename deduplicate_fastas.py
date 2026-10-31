#!/usr/bin/env python3
"""
deduplicate_fastas.py -- stage 2a of the Cats&Dogs pipeline.

Inside every species file, keep one record for each distinct sequence and drop
the other copies. Run it after remove_shared_records.py:

    python deduplicate_fastas.py felidae_clean felidae_dedup  --report felidae_dedup_report.csv
    python deduplicate_fastas.py canidae_clean canidae_dedup  --report canidae_dedup_report.csv

What counts as a duplicate
  Two records whose sequences are identical character by character after
  upper-casing and joining the lines (same length, same bases, same N's).
  Records that differ by one base, one N or one position of the origin are
  kept as distinct.

Which copy is kept
  A RefSeq record (accession starting with NC_) if the group has one,
  otherwise the record with the smallest accession (plain string order).

Order of the records in the output
  Sorted by accession (default), so that the seeded sampling in the analysis
  scripts does not depend on the order in which NCBI returned the records.
  Use --keep-order to keep the original order of the kept records instead.

The original files are never modified. The records are copied exactly as they
are in the input (header, line breaks, line endings). A file that has no
duplicates is copied unchanged, except for the order if --keep-order is not set.
Only the standard library is needed.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

EXTS = (".fasta", ".fa", ".fas")


def read_records(path: str) -> list[tuple[str, list[str]]]:
    """Return [(header_line, [sequence_lines])], lines kept exactly as in the file."""
    recs: list[tuple[str, list[str]]] = []
    head, body = None, []
    with open(path, newline="") as fh:
        for line in fh:
            if line.startswith(">"):
                if head is not None:
                    recs.append((head, body))
                head, body = line, []
            elif head is not None:
                body.append(line)
    if head is not None:
        recs.append((head, body))
    return recs


def accession(head: str) -> str:
    return head[1:].split()[0]


def sequence(body: list[str]) -> str:
    return "".join(l.strip() for l in body).upper()


def choose_representative(group: list[int], heads: list[str]) -> int:
    """Index (into the file's record list) of the copy to keep."""
    refseq = [i for i in group if accession(heads[i]).startswith("NC_")]
    pool = refseq or group
    return min(pool, key=lambda i: accession(heads[i]))


def write_record(out, head: str, body: list[str]) -> None:
    out.write(head)
    out.writelines(body)
    last = body[-1] if body else head
    if not last.endswith(("\n", "\r")):
        out.write("\r\n" if head.endswith("\r\n") else "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--report", help="CSV with one row per record: file, accession, action, kept_accession, group_size")
    ap.add_argument("--keep-order", action="store_true", help="keep the original order instead of sorting by accession")
    ap.add_argument("--dry-run", action="store_true", help="only print and write the report, no FASTA files")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(EXTS))
    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)

    report_rows = []
    total_in = total_out = 0
    for fname in files:
        recs = read_records(os.path.join(args.input_dir, fname))
        heads = [h for h, _ in recs]
        groups: dict[str, list[int]] = {}
        for i, (_, body) in enumerate(recs):
            groups.setdefault(sequence(body), []).append(i)

        keep: dict[int, int] = {}                 # kept index -> group size
        for g in groups.values():
            keep[choose_representative(g, heads)] = len(g)

        kept_acc_of = {}
        for g in groups.values():
            rep = choose_representative(g, heads)
            for i in g:
                kept_acc_of[i] = accession(heads[rep])

        for i, (h, _) in enumerate(recs):
            report_rows.append(dict(
                file=fname, accession=accession(h),
                action="kept" if i in keep else "removed",
                kept_accession=kept_acc_of[i],
                group_size=len(groups[sequence(recs[i][1])]),
            ))

        order = sorted(keep, key=lambda i: accession(heads[i])) if not args.keep_order else sorted(keep)
        total_in += len(recs)
        total_out += len(order)
        if len(order) != len(recs):
            print(f"{fname}: {len(recs)} -> {len(order)} records ({len(recs) - len(order)} identical copies removed)")

        if not args.dry_run:
            with open(os.path.join(args.output_dir, fname), "w", newline="") as out:
                for i in order:
                    write_record(out, *recs[i])

    print(f"\n{len(files)} files, {total_in} -> {total_out} records")
    if args.report:
        with open(args.report, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["file", "accession", "action", "kept_accession", "group_size"])
            w.writeheader()
            w.writerows(report_rows)
        print(f"report written to {args.report}")


if __name__ == "__main__":
    main()
