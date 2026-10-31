#!/usr/bin/env python3
"""
fetch_ncbi_mito_results.py

Fetch *every* NCBI Nucleotide result for a species' mitochondrial DNA search
(not just the handful a web search or a browser preview would show), and
return them as a list of strings, each with the record title and a direct
link, e.g.:

    "Panthera leo mitochondrion, complete genome — https://www.ncbi.nlm.nih.gov/nuccore/NC_028302.1"

Uses NCBI's public E-utilities API (esearch + esummary), which has no
authentication requirement but does have a rate limit: 3 requests/second
without an API key, 10/second with one (see --api-key / --email below).
Requires only the `requests` package (pip install requests).

Examples
--------
# Every "complete mitochondrial genome" result for the lion:
python fetch_ncbi_mito_results.py --species "Panthera leo"

# Only RefSeq (curated, NC_-prefixed) records:
python fetch_ncbi_mito_results.py --species "Panthera leo" --mode refseq

# Any mitochondrial record at all (partial genes included, not just complete genomes):
python fetch_ncbi_mito_results.py --species "Panthera leo" --mode any

# Run for every species in a text file (one scientific name per line),
# writing a CSV with one row per record:
python fetch_ncbi_mito_results.py --species-file species.txt --mode complete --out results.csv

# Also download the actual FASTA sequences, one combined multi-FASTA file per
# species, into a folder:
python fetch_ncbi_mito_results.py --species-file species.txt --mode complete --download-fasta fastas/

# Or call it as a library from your own code:
from fetch_ncbi_mito_results import fetch_mito_results, download_species_fastas
links = fetch_mito_results("Panthera leo", mode="complete")
for link in links:
    print(link)

path = download_species_fastas("Panthera leo", mode="complete", outdir="fastas")
print(f"Saved to {path}")
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from typing import Iterable

import requests

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DB = "nuccore"

# Same query patterns used to build the spreadsheet's search links.
MODE_SUFFIXES = {
    "complete": "AND mitochondrion[filter] AND complete genome[title]",
    "refseq": "AND mitochondrion[filter] AND srcdb_refseq[prop]",
    "any": "AND mitochondrion[filter]",
}


def build_term(species: str, mode: str = "complete") -> str:
    """Build the same Entrez query term used for the 'Complete mitogenome
    search' / 'RefSeq mitogenome only' / 'Any mitochondrial sequence' links."""
    if mode not in MODE_SUFFIXES:
        raise ValueError(f"mode must be one of {list(MODE_SUFFIXES)}, got {mode!r}")
    return f'"{species}"[Organism] {MODE_SUFFIXES[mode]}'


def _sleep_for_rate_limit(api_key: str | None) -> None:
    # NCBI: max 3 req/sec without an API key, 10 req/sec with one.
    time.sleep(0.11 if api_key else 0.35)


def esearch_all_ids(
    term: str,
    api_key: str | None = None,
    email: str | None = None,
    tool: str = "fetch_ncbi_mito_results",
    batch_size: int = 500,
    session: requests.Session | None = None,
) -> list[str]:
    """Return every UID matching an Entrez search term (paginates past the
    default result cap instead of returning just the first page)."""
    sess = session or requests.Session()
    ids: list[str] = []
    retstart = 0
    while True:
        params = {
            "db": DB,
            "term": term,
            "retstart": retstart,
            "retmax": batch_size,
            "retmode": "json",
            "tool": tool,
        }
        if api_key:
            params["api_key"] = api_key
        if email:
            params["email"] = email

        resp = sess.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=30)
        resp.raise_for_status()
        result = resp.json()["esearchresult"]

        if "ERROR" in result:
            raise RuntimeError(f"NCBI esearch error for term {term!r}: {result['ERROR']}")

        count = int(result.get("count", 0))
        batch_ids = result.get("idlist", [])
        ids.extend(batch_ids)

        retstart += batch_size
        _sleep_for_rate_limit(api_key)

        if retstart >= count or not batch_ids:
            break

    return ids


def esummary_records(
    ids: Iterable[str],
    api_key: str | None = None,
    email: str | None = None,
    tool: str = "fetch_ncbi_mito_results",
    batch_size: int = 200,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch accession + title for each UID, in batches."""
    sess = session or requests.Session()
    ids = list(ids)
    records: list[dict] = []

    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        params = {
            "db": DB,
            "id": ",".join(batch),
            "retmode": "json",
            "tool": tool,
        }
        if api_key:
            params["api_key"] = api_key
        if email:
            params["email"] = email

        resp = sess.get(f"{EUTILS_BASE}/esummary.fcgi", params=params, timeout=30)
        resp.raise_for_status()
        result = resp.json()["result"]

        for uid in result.get("uids", []):
            doc = result[uid]
            accession = doc.get("accessionversion") or doc.get("caption") or uid
            records.append(
                {
                    "uid": uid,
                    "accession": accession,
                    "title": doc.get("title", ""),
                    "organism": doc.get("organism", ""),
                }
            )

        _sleep_for_rate_limit(api_key)

    return records


def _split_fasta(text: str) -> list[str]:
    """Split a multi-FASTA text blob (as returned by efetch) into a list of
    individual '>...' record strings."""
    records: list[str] = []
    current: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith(">"):
            if current:
                records.append("".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        records.append("".join(current))
    return records


def fetch_fasta(
    accessions: Iterable[str],
    api_key: str | None = None,
    email: str | None = None,
    tool: str = "fetch_ncbi_mito_results",
    batch_size: int = 200,
    session: requests.Session | None = None,
) -> dict[str, str]:
    """Download the actual FASTA sequence text for a list of accessions
    (e.g. ["NC_028302.1", "KF907306.1"]), via efetch.

    Returns {accession_as_in_header: fasta_text}, where fasta_text is that
    one record's full '>header\\nSEQUENCE...' text. Uses POST (not GET) so
    large batches don't hit URL-length limits, and batches requests to stay
    well under NCBI's per-call size recommendations.
    """
    sess = session or requests.Session()
    accessions = list(accessions)
    out: dict[str, str] = {}

    for i in range(0, len(accessions), batch_size):
        batch = accessions[i : i + batch_size]
        data = {
            "db": DB,
            "id": ",".join(batch),
            "rettype": "fasta",
            "retmode": "text",
            "tool": tool,
        }
        if api_key:
            data["api_key"] = api_key
        if email:
            data["email"] = email

        resp = sess.post(f"{EUTILS_BASE}/efetch.fcgi", data=data, timeout=60)
        resp.raise_for_status()

        for rec_text in _split_fasta(resp.text):
            header = rec_text.splitlines()[0]
            acc_in_header = header[1:].split()[0]  # first token after '>'
            out[acc_in_header] = rec_text

        _sleep_for_rate_limit(api_key)

    return out


def download_species_fastas(
    species: str,
    mode: str = "complete",
    outdir: str = ".",
    combine: bool = True,
    api_key: str | None = None,
    email: str | None = None,
) -> list[str]:
    """Search + download FASTAs for one species in one call, and write them
    to disk. Returns the list of file path(s) written.

    combine=True (default): one multi-FASTA file per species, e.g.
        outdir/Panthera_leo_complete.fasta
    combine=False: one file per accession, e.g.
        outdir/Panthera_leo/NC_028302.1.fasta
    """
    import os

    term = build_term(species, mode)
    ids = esearch_all_ids(term, api_key=api_key, email=email)
    if not ids:
        return []

    records = esummary_records(ids, api_key=api_key, email=email)
    accessions = [r["accession"] for r in records]
    fastas = fetch_fasta(accessions, api_key=api_key, email=email)

    species_slug = species.replace(" ", "_")
    os.makedirs(outdir, exist_ok=True)
    written: list[str] = []

    if combine:
        path = os.path.join(outdir, f"{species_slug}_{mode}.fasta")
        with open(path, "w", encoding="utf-8") as f:
            for acc in accessions:
                if acc in fastas:
                    f.write(fastas[acc])
                    if not fastas[acc].endswith("\n"):
                        f.write("\n")
        written.append(path)
    else:
        sp_dir = os.path.join(outdir, species_slug)
        os.makedirs(sp_dir, exist_ok=True)
        for acc, fasta_text in fastas.items():
            path = os.path.join(sp_dir, f"{acc}.fasta")
            with open(path, "w", encoding="utf-8") as f:
                f.write(fasta_text)
            written.append(path)

    return written


def fetch_mito_results(
    species: str,
    mode: str = "complete",
    api_key: str | None = None,
    email: str | None = None,
) -> list[str]:
    """Fetch ALL matching records for one species and return them as a list
    of 'Title — link' strings, e.g.:

        'Panthera leo persica mitochondrion, complete genome — https://www.ncbi.nlm.nih.gov/nuccore/NC_018053.1'

    mode: "complete" (complete mitochondrial genomes only, the default),
          "refseq" (curated RefSeq/NC_ records only), or
          "any" (every mitochondrial record, including partial genes).
    """
    term = build_term(species, mode)
    ids = esearch_all_ids(term, api_key=api_key, email=email)
    records = esummary_records(ids, api_key=api_key, email=email)

    links = []
    for r in records:
        url = f"https://www.ncbi.nlm.nih.gov/nuccore/{r['accession']}"
        title = r["title"] or r["accession"]
        links.append(f"{title} — {url}")
    return links


def fetch_mito_results_multi(
    species_list: Iterable[str],
    mode: str = "complete",
    api_key: str | None = None,
    email: str | None = None,
) -> dict[str, list[str]]:
    """Same as fetch_mito_results, but for many species at once. Returns
    {species_name: [list of 'Title — link' strings]}."""
    out = {}
    for sp in species_list:
        out[sp] = fetch_mito_results(sp, mode=mode, api_key=api_key, email=email)
    return out


def _read_species_file(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--species", help='Scientific name, e.g. "Panthera leo"')
    group.add_argument("--species-file", help="Text file with one scientific name per line")
    parser.add_argument(
        "--mode",
        choices=["complete", "refseq", "any"],
        default="complete",
        help="complete = complete mitochondrial genomes only (default); "
        "refseq = curated RefSeq (NC_) records only; any = every mitochondrial record",
    )
    parser.add_argument("--api-key", default=None, help="NCBI API key (optional; raises the rate limit to 10 req/sec)")
    parser.add_argument("--email", default=None, help="Contact email (optional, but polite/recommended by NCBI)")
    parser.add_argument("--out", default=None, help="Write results to this CSV file instead of just printing them")
    parser.add_argument(
        "--download-fasta",
        metavar="DIR",
        default=None,
        help="Also download the actual FASTA sequences into this folder "
        "(one combined multi-FASTA file per species, unless --no-combine is set)",
    )
    parser.add_argument(
        "--no-combine",
        action="store_true",
        help="With --download-fasta, write one file per accession instead of one combined file per species",
    )
    args = parser.parse_args()

    species_list = [args.species] if args.species else _read_species_file(args.species_file)

    all_rows = []  # (species, link_string) for CSV output
    for sp in species_list:
        print(f"\n=== {sp} ({args.mode}) ===", file=sys.stderr)
        try:
            links = fetch_mito_results(sp, mode=args.mode, api_key=args.api_key, email=args.email)
        except requests.RequestException as exc:
            print(f"  ERROR fetching {sp}: {exc}", file=sys.stderr)
            continue

        print(f"  {len(links)} result(s) found", file=sys.stderr)
        for link in links:
            print(link)
            all_rows.append((sp, link))

        if args.download_fasta and links:
            try:
                paths = download_species_fastas(
                    sp,
                    mode=args.mode,
                    outdir=args.download_fasta,
                    combine=not args.no_combine,
                    api_key=args.api_key,
                    email=args.email,
                )
                for p in paths:
                    print(f"  wrote {p}", file=sys.stderr)
            except requests.RequestException as exc:
                print(f"  ERROR downloading FASTA for {sp}: {exc}", file=sys.stderr)

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["species", "result"])
            writer.writerows(all_rows)
        print(f"\nWrote {len(all_rows)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
