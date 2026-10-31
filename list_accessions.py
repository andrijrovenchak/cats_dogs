"""List the accession numbers of ALL records in the deduplicated folders and mark the (at most 3 per species file) random records used in the analyses.

The two kinds of analyses in the paper draw the records differently, because mtdna.py seeds the random generator differently:
  A  classification, SVD scan, word weights, baselines:  get_tokenized_documents() calls random.seed(42) before sampling
     EACH file, so every file is sampled independently of the others.
  B  distance matrices and trees (Jaccard, K2P, Euclid, k-mers):  build_*_matrix() calls random.seed(42) ONCE and then samples
     the files one after another in sorted order, so a file's sample depends on the files before it.
Both draw with random.sample(records, 3) from the records in file order, and only when a file has more than 3 records.

Usage: python list_accessions.py [felidae_dir] [canidae_dir]      (default: felidae_dedup canidae_dedup)
Output: accessions_all.csv (one row per record in the folders; accession without version, version, length in bp, Y/N for non-ACGT symbols with their count, percentage and the number of N, and 0/1 flags for the records sampled in A and in B) and a printed summary."""
import sys, os, glob, random, csv, re
from Bio import SeqIO

DIRS = (('Felidae', sys.argv[1] if len(sys.argv) > 1 else 'felidae_dedup'),
        ('Canidae', sys.argv[2] if len(sys.argv) > 2 else 'canidae_dedup'))
MAX = 3

def read(path):
    return list(SeqIO.parse(path, 'fasta'))
def acc(rec):                       # '>NC_012345.1 Panthera leo ...' -> 'NC_012345.1'
    return rec.description.split()[0]
def shown(rec):                     # (accession, version); records that are not GenBank/RefSeq entries get a label
    a = acc(rec)
    if re.fullmatch(r'[A-Z]{1,2}_?\d+(\.\d+)?', a):
        return (a.split('.')[0], a.split('.')[1] if '.' in a else '')
    return ('not in INSDC (assembled in this study, SRR35083483)', '')
def species(path):
    return os.path.basename(path).replace('_complete.fasta', '').replace('.fasta', '')

rows = []
for family, d in DIRS:
    files = sorted(glob.glob(os.path.join(d, '*.fasta')))
    # scheme B: one seed, files in sorted order
    random.seed(42)
    chosen_b = {}
    for f in files:
        recs = read(f)
        chosen_b[f] = {acc(r) for r in (random.sample(recs, MAX) if len(recs) > MAX else recs)}
    # scheme A: seed before every file; then write ALL records with the two flags
    for f in files:
        recs = read(f)
        if len(recs) > MAX:
            random.seed(42); chosen_a = {acc(r) for r in random.sample(recs, MAX)}
        else:
            chosen_a = {acc(r) for r in recs}
        for r in recs:
            rows.append({'family': family, 'species': species(f), 'records_in_file': len(recs),
                         'accession': shown(r)[0], 'version': shown(r)[1],
                         'length_bp': len(r.seq), 'non_ACGT': 'Y' if set(str(r.seq).upper()) - set('ACGT') else 'N',
                         'non_ACGT_count': sum(ch not in 'ACGT' for ch in str(r.seq).upper()),
                         'non_ACGT_percent': round(100 * sum(ch not in 'ACGT' for ch in str(r.seq).upper()) / max(len(r.seq), 1), 3),
                         'N_count': str(r.seq).upper().count('N'),
                         'sampled_A_classification': int(acc(r) in chosen_a),
                         'sampled_B_trees': int(acc(r) in chosen_b[f]), 'header': r.description})

with open('accessions_all.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

for family, _ in DIRS:
    r = [x for x in rows if x['family'] == family]
    print(f"{family}: {len(r)} records from {len({x['species'] for x in r})} species; "
          f"sampled in A: {sum(x['sampled_A_classification'] for x in r)}, in B: {sum(x['sampled_B_trees'] for x in r)}; "
          f"with non-ACGT symbols: {sum(x['non_ACGT'] == 'Y' for x in r)} (in A: {sum(x['non_ACGT'] == 'Y' and x['sampled_A_classification'] for x in r)})")
dup = len(rows) - len({(x['accession'], x['version']) for x in rows})
print(f"accessions listed more than once (across species files or families): {dup}")
print('written: accessions_all.csv')
