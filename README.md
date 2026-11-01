# Cats&Dogs pipeline
 
Alignment-free classification of Felidae and Canidae mitogenomes with adenine-delimited
words, and comparison of UPGMA species trees with literature reference trees.
Code for the paper "Of Cats and Dogs: What Languages Do Mitochondrions Speak? A Vector-Space Analysis of mtDNA Vocabulary". 
Archived version: <Zenodo DOI>.
 
## Requirements
 
Python 3.10+; `numpy`, `pandas` (< 3.0, see Notes), `scipy`, `scikit-learn`, `biopython`,
`openpyxl`, `matplotlib`, `ete3`. Exact versions: `requirements.txt`.
External programs (only for the maximum-likelihood trees): MAFFT v7.505, IQ-TREE 2.0.7.
 
Run every script in the folder that contains `mtdna.py` and the FASTA folders
(`mtdna.py` is imported by most of the others). Seeds are fixed (42 in the Python code, 1 in IQ-TREE).
 
## Folder and file names
 
| Name | Content |
|---|---|
| `felidae_fastas/`, `canidae_fastas/` | raw downloads, one multi-FASTA per species (`Genus_species_complete.fasta`) |
| `felidae_clean/`, `canidae_clean/` | after removing records shared with a subspecies or split species |
| `felidae_dedup/`, `canidae_dedup/` | after removing identical sequences within a species; **input of all analyses** |
| `{family}_{distance}_matrix[-k13].csv` | species-by-species distance matrices |
| `{family}{n if canidae}_{method}_tree.nwk` | UPGMA trees (`canidae29_…` with *L. adusta*, `canidae28_…` without) |
| `reference_trees.nwk` | reference topologies (headers are the keys used by `tanglegram.py`) |
| `supplementary.pdf` | supplemetary matrial for the manuscript |
 
## Steps
 
### 1. Download the records
    python collect_fastas.py
Reads `felidae.xlsx` and `canidae.xlsx` (sheet "… mtDNA links", column "Scientific name") and
downloads all complete mitochondrial genomes of every species with
`fetch_ncbi_mito_results.download_species_fastas` into `felidae_fastas/`, `canidae_fastas/`.
The NCBI query date matters: the original queries were run on 27 September 2026 for Felidae and 5 October 2026 for Canidae. 
A new run returns the records that exist at that time, so it can differ from the collection used here
(new submissions, changed records). To reproduce the results, start from the `*_dedup` folders in this
repository (step 3 onward) or check a new download against `accessions_all.csv`.
 
### 1a. Manual lookup
Do manual lookup for species with empty FASTA sets, like
`Leopardus emiliae -> []` or `Vulpes bengalensis -> []`.
E.g., check literature, look for other possible species names, etc.
Consider searching almost complete mitochondrions as well if applicable.
 
*L. tilcayo* (Felidae): the mitogenome is not in GenBank and was assembled from reads
(see `notebooks/extract_mtdna_SRX30196846.ipynb`, Appendix A of the paper); its FASTA is added to the Felidae folder by hand.
 
### 2. Remove shared FASTA
    python remove_shared_records.py felidae_fastas felidae_clean
    python remove_shared_records.py canidae_fastas canidae_clean
These might occur due to species/subspecies ambiguity, like *Canis lupus familiaris* (treating
the domestic dog as a subspecies), which will end up in both *Canis lupus* and *Canis familiaris*.
The script removes from the parent file (e.g. *Canis lupus*) the accessions that are also in the
file of a child taxon (*C. familiaris*, *C. lycaon*, *F. bieti*, *F. lybica*, *L. emiliae*);
edit `PARENT_OF` to change this. Originals are never changed.
 
### 2a. Remove identical sequences
    python deduplicate_fastas.py felidae_clean felidae_dedup --report felidae_dedup_report.csv
    python deduplicate_fastas.py canidae_clean canidae_dedup --report canidae_dedup_report.csv
Keeps one record per distinct sequence per species (RefSeq first, then the smallest accession).
 
### 3. Summary and tables
    python summarize_fastas.py felidae_dedup canidae_dedup --out summary --verify-mtdna .
    python list_accessions.py felidae_dedup canidae_dedup
The first script writes `records.csv`, `species.csv`, `accessions_supplement.csv`, the LaTeX tables and
`checks.txt` (shared accessions, identical sequences across species, strand flags, odd lengths).
The second script writes `accessions_all.csv`: all records with length, share of non-ACGT characters
and the two samples (classification and distance trees); it is the source of Supplementary Table S1.
Run the scripts on the same folders that the analyses use (`*_dedup`).

### 4. Family and genus classification
    python classify_families.py          # Felidae vs Canidae, linear SVM on TF-IDF words
    python classify_families_lsa.py      # same with truncated SVD (66 components)
    python classify_genera.py            # Vulpes vs Canis
    python check_fastas.py               # sanity checks of the corpora
Tokens are the words between adenines (empty word = `O`), windows of 150 words every 75 words,
at most 3 records per species. Binary TF-IDF of unigrams and bigrams (`min_df=3`), linear SVM
(`C=1`, balanced class weights), 5-fold `GroupKFold` with the species file as the group,
so that no species is in both training and test folds.
 
### 5. Distance matrices and UPGMA trees
    python cats_dogs.py
The flag `XLS_BUILT` at the top selects whether the matrices are computed 
(`False`, slow: K2P uses global pairwise alignment) or read from the existing csv files (`True`).
Set `XLS_BUILT = False` for the first run, which computes the matrices (note that K2P is slow and can run for several hours). 
The script writes the `*_tree.nwk` files into the working directory, not `trees/`.
For every family it builds Jaccard and Mash (13-mers), K2P, Euclidean (genomes), TF-IDF cosine,
cosine and Euclidean on species-level vectors, builds the UPGMA trees, prints the Robinson-Foulds
distance to the reference tree, and writes `*_tree.nwk`.
    python check_jaccard_kmers.py        # RF distance as a function of k (3-21)
    python pretty_newick.py in.nwk out.nwk   # drop the unary root, tidy format
 
### 6. Figures
    python plot_distance_heatmaps.py [--suffix -28]
    python tanglegram.py felidae --folder trees --reference trees/reference_trees.nwk                 # lowest-RF tree against the reference
    python tanglegram.py canidae 28 --method tfidf --folder trees --reference trees/reference_trees.nwk
 
### 7. Maximum-likelihood trees (MAFFT + IQ-TREE)
Run `notebooks/MAFFT_IQTREE.ipynb` in Google Colab (Ubuntu image; MAFFT and IQ-TREE come from `apt`).
Inputs, packed into `fastas_pack.zip`: the folders `felidae_dedup` and `canidae_dedup`,
`export_for_mafft.py`, and two outgroup records saved as `hyena.fasta` (spotted hyena,
*Crocuta crocuta*, NC_020669.1) and `bear.fasta` (brown bear, *Ursus arctos*, NC_003427.1).
The notebook
1. rotates one mitogenome per species to a common origin (`export_for_mafft.py`; anchors
   MW257207.1 for Felidae, NC_002008.4 for Canidae; *L. adusta* is dropped from Canidae, hence "canidae28"),
2. aligns with `mafft --auto`,
3. runs `iqtree2 -m GTR+G -bb 1000 -nt AUTO -seed 1` (outgroups *C. crocuta* and *U. arctos*),
4. removes the outgroup and writes `felidae_mt_ml.nwk`, `canidae28_mt_ml.nwk` (topology only) and
   `*_mt_ml_support.nwk` (IQ-TREE tree with UFBoot support), and packs the alignment, report, log,
   trees and `*_mt_records.tsv` into `felidae_ml.zip`, `canidae28_ml.zip`.
`-nt AUTO` lets IQ-TREE choose the number of threads by timing, which can change the UFBoot values
slightly (not the topology). Use `-nt 1` for identical support values on every run.
 
### 8. *L. tilcayo* mitogenome from reads
Run `notebooks/extract_mtdna_SRX30196846.ipynb` (Colab; SRX30196846 = SRR35083483, GetOrganelle
`animal_mt`). The 17,104 bp circular sequence has no GenBank accession; save it as
`Leopardus_tilcayo_complete.fasta` in the Felidae folder (step 1a).

## Notes
 
* pandas 3 (Copy-on-Write): `np.fill_diagonal(df.values, …)` in `mtdna.py` fails; use pandas < 3
  or copy the array before filling (`arr = df.to_numpy(copy=True); np.fill_diagonal(arr, 0.0)`).
* Distances between species: mean of the distances between sampled records; the diagonal is the
  largest within-species distance.
* Only the topology of the UPGMA trees is used.
