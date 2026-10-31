"""Remove records from the broader species file when the same accession is also
present in the file of a more specific taxon (subspecies or split species).

Usage:  python remove_shared_records.py INPUT_DIR OUTPUT_DIR [--dry-run]
Originals are never modified; cleaned copies are written to OUTPUT_DIR.
Edit PARENT_OF to change which file loses the shared records.
"""
import os, sys, collections

# child file -> parent file (the parent loses the record)
PARENT_OF = {
    'Canis_familiaris_complete.fasta':  'Canis_lupus_complete.fasta',
    'Canis_lycaon_complete.fasta':      'Canis_lupus_complete.fasta',
    'Felis_bieti_complete.fasta':       'Felis_silvestris_complete.fasta',
    'Felis_lybica_complete.fasta':      'Felis_silvestris_complete.fasta',
    'Leopardus_emiliae_complete.fasta': 'Leopardus_tigrinus_complete.fasta',
}

def read_records(path):
    recs, head, body = [], None, []
    with open(path, newline='') as fh:
        for line in fh:
            if line.startswith('>'):
                if head is not None: recs.append((head, body))
                head, body = line, []
            elif head is not None:
                body.append(line)
    if head is not None: recs.append((head, body))
    return recs

def acc(head): return head[1:].split()[0]

def main(src, dst, dry=False):
    files = sorted(f for f in os.listdir(src) if f.endswith(('.fasta', '.fa', '.fas')))
    child_acc = collections.defaultdict(set)          # parent file -> accessions held by its children
    for child, parent in PARENT_OF.items():
        if child in files and parent in files:
            child_acc[parent] |= {acc(h) for h, _ in read_records(os.path.join(src, child))}
    if not dry: os.makedirs(dst, exist_ok=True)
    for f in files:
        recs = read_records(os.path.join(src, f))
        kept = [(h, b) for h, b in recs if acc(h) not in child_acc.get(f, set())]
        if len(kept) != len(recs):
            print(f'{f}: {len(recs)} -> {len(kept)} records ({len(recs)-len(kept)} removed)')
        if not dry:
            with open(os.path.join(dst, f), 'w', newline='') as out:
                for h, b in kept:
                    out.write(h); out.writelines(b)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], '--dry-run' in sys.argv)