import math
import random
import os
import glob
import math
import itertools

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch

from io import StringIO
from Bio import SeqIO, Align, Phylo
from ete3 import Tree

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import normalize

from scipy.spatial.distance import squareform, pdist, euclidean, cosine
from scipy.cluster.hierarchy import linkage



def get_kmers(fasta_path: str, k: int = 21):
    """
    Extracts a set of all contiguous k-mers from a raw FASTA sequence.
    """
    record = next(SeqIO.parse(fasta_path, "fasta"))
    seq = str(record.seq).upper()
    
    # Slide a window of length k across the raw string one base at a time.
    # E.g., for "GATTACA" with k=3 -> {"GAT", "ATT", "TTA", "TAC", "ACA"}
    kmers = set([seq[i:i+k] for i in range(len(seq) - k + 1)])
    
    return kmers


def compute_raw_jaccard_distance(fasta1: str, fasta2: str, k: int = 21):
    """
    Calculates Jaccard distance: 1 - (Intersection / Union) on raw k-mer sets.
    """
    kmers1 = get_kmers(fasta1, k)
    kmers2 = get_kmers(fasta2, k)
    
    intersection = len(kmers1.intersection(kmers2))
    union = len(kmers1.union(kmers2))
    
    if union == 0:
        return 1.0 # Edge case fallback
        
    jaccard_distance = 1.0 - (intersection / union)
    
    return jaccard_distance


def get_kmers_from_sequence(seq: str, k: int = 21):
    """
    Extracts a set of all contiguous k-mers directly from a sequence string 
    (bypassing file path parsing).
    """
    # Slide a window of length k across the raw string
    kmers = set([seq[i:i+k] for i in range(len(seq) - k + 1)])
    return kmers


def compute_raw_jaccard_distance_from_seqs(seq1: str, seq2: str, k: int = 21):
    """
    Calculates Jaccard distance directly from two sequence strings.
    """
    kmers1 = get_kmers_from_sequence(seq1, k)
    kmers2 = get_kmers_from_sequence(seq2, k)
    
    intersection = len(kmers1.intersection(kmers2))
    union = len(kmers1.union(kmers2))
    
    if union == 0:
        return 1.0
        
    return 1.0 - (intersection / union)


def build_population_jaccard_matrix(family: str, directory_path: str, file_extension: str = "*.fasta", max_seqs: int = 2, seed: int = 42, k: int = 21):
    """Generates an inter/intra-species distance matrix from a folder of multi-FASTA files."""
    
    # Lock the random seed so subsampling yields the exact same 6 sequences on repeat runs
    random.seed(seed)
    
    search_pattern = os.path.join(directory_path, file_extension)
    fasta_files = sorted(glob.glob(search_pattern))
    
    if not fasta_files:
        print("No FASTA files found.")
        return None

    clades = {}
    file_labels = []
    
    for file in fasta_files:
        filename = os.path.basename(file)
        file_labels.append(filename)
        
        records = list(SeqIO.parse(file, "fasta"))
        raw_seqs = [str(rec.seq).upper() for rec in records]
        
        # Check and apply the random subsampling limit
        if len(raw_seqs) > max_seqs:
            print(f"[{filename}] Found {len(raw_seqs)} sequences. Randomly subsampling to {max_seqs}.")
            clades[filename] = random.sample(raw_seqs, max_seqs)
        else:
            clades[filename] = raw_seqs
            
    n = len(file_labels)
    dist_matrix = np.zeros((n, n))
    
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1
    
    print(f"Loaded {n} files.")

    # 1. Calculate Diagonal (Intra-species Maximum)
    for i in range(n):
        label = file_labels[i]
        seqs = clades[label]
        
        if len(seqs) <= 1:
            dist_matrix[i, i] = 0.0
            print(f"[{label}] 1 sequence: Intra-species distance = 0.0")
        else:
            intra_distances = []
            # Compare every sequence against every other sequence WITHIN the same file
            for s1, s2 in itertools.combinations(seqs, 2):
                intra_distances.append(compute_raw_jaccard_distance_from_seqs(s1, s2, k=k))
            
            # Using np.nanmax safely handles math errors if certain intra-seqs are too divergent
            max_dist = np.nanmax(intra_distances)
            dist_matrix[i, i] = max_dist
            print(f"[{label}] {len(seqs)} sequences: Max Intra-species distance = {max_dist:.5f}")

    # 2. Calculate Off-Diagonal (Inter-species Average)
    for i, j in itertools.combinations(range(n), 2):
        print(f'\ri, j = {i} {j}', end='')
        label_A = file_labels[i]
        label_B = file_labels[j]
        seqs_A = clades[label_A]
        seqs_B = clades[label_B]
        
        inter_distances = []
        # Compare EVERY sequence in File A to EVERY sequence in File B (Cartesian product)
        for sA, sB in itertools.product(seqs_A, seqs_B):
            inter_distances.append(compute_raw_jaccard_distance_from_seqs(sA, sB, k=k))
            
        # The standard metric for inter-clade distance is the average pairwise distance
        mean_dist = np.nanmean(inter_distances)
        
        # Mirror the matrix
        dist_matrix[i, j] = mean_dist
        dist_matrix[j, i] = mean_dist
        
    # 3. Export to DataFrame
    df = pd.DataFrame(dist_matrix, index=file_labels, columns=file_labels)
    df.to_csv(f"{family}_jaccard_matrix-k{k}.csv")
    print("\nMatrix successfully saved to population_jaccard_matrix.csv")
    
    return df


def compute_mash_distance_from_seqs(seq1: str, seq2: str, k: int = 21):
    j = 1.0 - compute_raw_jaccard_distance_from_seqs(seq1, seq2, k)
    if j <= 0:
        return 1.0                       # no shared k-mers
    return max(0.0, -math.log(2 * j / (1 + j)) / k)


def build_population_mash_matrix(family: str, directory_path: str, file_extension: str = "*.fasta", max_seqs: int = 2, seed: int = 42, k: int = 21):
    """Generates an inter/intra-species distance matrix from a folder of multi-FASTA files."""
    
    # Lock the random seed so subsampling yields the exact same 6 sequences on repeat runs
    random.seed(seed)
    
    search_pattern = os.path.join(directory_path, file_extension)
    fasta_files = sorted(glob.glob(search_pattern))
    
    if not fasta_files:
        print("No FASTA files found.")
        return None

    clades = {}
    file_labels = []
    
    for file in fasta_files:
        filename = os.path.basename(file)
        file_labels.append(filename)
        
        records = list(SeqIO.parse(file, "fasta"))
        raw_seqs = [str(rec.seq).upper() for rec in records]
        
        # Check and apply the random subsampling limit
        if len(raw_seqs) > max_seqs:
            print(f"[{filename}] Found {len(raw_seqs)} sequences. Randomly subsampling to {max_seqs}.")
            clades[filename] = random.sample(raw_seqs, max_seqs)
        else:
            clades[filename] = raw_seqs
            
    n = len(file_labels)
    dist_matrix = np.zeros((n, n))
    
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1
    
    print(f"Loaded {n} files.")

    # 1. Calculate Diagonal (Intra-species Maximum)
    for i in range(n):
        label = file_labels[i]
        seqs = clades[label]
        
        if len(seqs) <= 1:
            dist_matrix[i, i] = 0.0
            print(f"[{label}] 1 sequence: Intra-species distance = 0.0")
        else:
            intra_distances = []
            # Compare every sequence against every other sequence WITHIN the same file
            for s1, s2 in itertools.combinations(seqs, 2):
                intra_distances.append(compute_mash_distance_from_seqs(s1, s2, k=k))
            
            # Using np.nanmax safely handles math errors if certain intra-seqs are too divergent
            max_dist = np.nanmax(intra_distances)
            dist_matrix[i, i] = max_dist
            print(f"[{label}] {len(seqs)} sequences: Max Intra-species distance = {max_dist:.5f}")

    # 2. Calculate Off-Diagonal (Inter-species Average)
    for i, j in itertools.combinations(range(n), 2):
        print(f'\ri, j = {i} {j}', end='')
        label_A = file_labels[i]
        label_B = file_labels[j]
        seqs_A = clades[label_A]
        seqs_B = clades[label_B]
        
        inter_distances = []
        # Compare EVERY sequence in File A to EVERY sequence in File B (Cartesian product)
        for sA, sB in itertools.product(seqs_A, seqs_B):
            inter_distances.append(compute_mash_distance_from_seqs(sA, sB, k=k))
            
        # The standard metric for inter-clade distance is the average pairwise distance
        mean_dist = np.nanmean(inter_distances)
        
        # Mirror the matrix
        dist_matrix[i, j] = mean_dist
        dist_matrix[j, i] = mean_dist
        
    # 3. Export to DataFrame
    df = pd.DataFrame(dist_matrix, index=file_labels, columns=file_labels)
    df.to_csv(f"{family}_mash_matrix-k{k}.csv")
    print("\nMatrix successfully saved to population_mash_matrix.csv")
    
    return df
    

def calc_k2p_core(seq1: str, seq2: str, aligner: Align.PairwiseAligner) -> float:
    """Core function to calculate K2P distance between two sequence strings."""
    best_alignment = aligner.align(seq1, seq2)[0]
    aligned_seq1, aligned_seq2 = best_alignment[0], best_alignment[1]

    transitions, transversions, valid_length = 0, 0, 0
    transition_pairs = [{'A', 'G'}, {'C', 'T'}]

    for b1, b2 in zip(aligned_seq1, aligned_seq2):
        if b1 == '-' or b2 == '-':
            continue
        valid_length += 1
        if b1 != b2:
            if {b1, b2} in transition_pairs:
                transitions += 1
            else:
                transversions += 1

    if valid_length == 0:
        return float('nan')

    p = transitions / valid_length
    q = transversions / valid_length

    try:
        term1 = 1.0 - 2.0 * p - q
        term2 = 1.0 - 2.0 * q
        if term1 <= 0 or term2 <= 0:
            return float('inf')
        return abs(-0.5 * math.log(term1) - 0.25 * math.log(term2))
    except ValueError:
        return float('nan')


# Added max_seqs and seed parameters
def build_population_k2p_matrix(family: str, directory_path: str, file_extension: str = "*.fasta", max_seqs: int = 2, seed: int = 42):
    """Generates an inter/intra-species distance matrix from a folder of multi-FASTA files."""

    # Lock the random seed so subsampling yields the exact same 6 sequences on repeat runs
    random.seed(seed)

    search_pattern = os.path.join(directory_path, file_extension)
    fasta_files = sorted(glob.glob(search_pattern))

    if not fasta_files:
        print("No FASTA files found.")
        return None

    clades = {}
    file_labels = []

    for file in fasta_files:
        filename = os.path.basename(file)
        file_labels.append(filename)

        records = list(SeqIO.parse(file, "fasta"))
        raw_seqs = [str(rec.seq).upper() for rec in records]

        # Check and apply the random subsampling limit
        if len(raw_seqs) > max_seqs:
            print(f"[{filename}] Found {len(raw_seqs)} sequences. Randomly subsampling to {max_seqs}.")
            clades[filename] = random.sample(raw_seqs, max_seqs)
        else:
            clades[filename] = raw_seqs

    n = len(file_labels)
    dist_matrix = np.zeros((n, n))

    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1

    print(f"Loaded {n} files.")

    # 1. Calculate Diagonal (Intra-species Maximum)
    for i in range(n):
        label = file_labels[i]
        seqs = clades[label]

        if len(seqs) <= 1:
            dist_matrix[i, i] = 0.0
            print(f"[{label}] 1 sequence: Intra-species distance = 0.0")
        else:
            intra_distances = []
            # Compare every sequence against every other sequence WITHIN the same file
            for s1, s2 in itertools.combinations(seqs, 2):
                intra_distances.append(calc_k2p_core(s1, s2, aligner))

            # Using np.nanmax safely handles math errors if certain intra-seqs are too divergent
            max_dist = np.nanmax(intra_distances)
            dist_matrix[i, i] = max_dist
            print(f"[{label}] {len(seqs)} sequences: Max Intra-species distance = {max_dist:.5f}")

    # 2. Calculate Off-Diagonal (Inter-species Average)
    for i, j in itertools.combinations(range(n), 2):
        print(f"\r\033[K- Processing i, j = {i} {j}", end="", flush=True)
        label_A = file_labels[i]
        label_B = file_labels[j]
        seqs_A = clades[label_A]
        seqs_B = clades[label_B]

        inter_distances = []
        # Compare EVERY sequence in File A to EVERY sequence in File B (Cartesian product)
        for sA, sB in itertools.product(seqs_A, seqs_B):
            inter_distances.append(calc_k2p_core(sA, sB, aligner))

        # The standard metric for inter-clade distance is the average pairwise distance
        mean_dist = np.nanmean(inter_distances)

        # Mirror the matrix
        dist_matrix[i, j] = mean_dist
        dist_matrix[j, i] = mean_dist

    # 3. Export to DataFrame
    df = pd.DataFrame(dist_matrix, index=file_labels, columns=file_labels)
    df.to_csv(f"{family}_k2p_matrix.csv")
    print("\nMatrix successfully saved to population_k2p_matrix.csv")

    return df


def update_population_k2p_matrix(directory_path: str, matrix_csv: str = "population_k2p_matrix.csv", file_extension: str = "*.fasta", max_seqs: int = 2, seed: int = 42):
    """Updates an existing K2P distance matrix with new FASTA files without recalculating old pairs."""
    
    # 1. Load the existing matrix
    if not os.path.exists(matrix_csv):
        print(f"'{matrix_csv}' not found. Please run the full build function first.")
        return None
        
    df_existing = pd.read_csv(matrix_csv, index_col=0)
    existing_labels = df_existing.index.tolist()
    
    # 2. Get current directory contents
    search_pattern = os.path.join(directory_path, file_extension)
    fasta_files = sorted(glob.glob(search_pattern))
    all_labels = [os.path.basename(f) for f in fasta_files]
    
    # Identify new files and valid existing files (in case any old files were deleted)
    new_labels = [l for l in all_labels if l not in existing_labels]
    valid_existing = [l for l in existing_labels if l in all_labels]
    
    if not new_labels:
        print("No new FASTA files found. Matrix is already up to date.")
        return df_existing

    print(f"Found {len(new_labels)} new files. Expanding matrix...")

    # 3. Load sequences (Must run through ALL files in order to maintain the exact random seed state)
    random.seed(seed)
    clades = {}
    
    for file in fasta_files:
        filename = os.path.basename(file)
        records = list(SeqIO.parse(file, "fasta"))
        raw_seqs = [str(rec.seq).upper() for rec in records]

        if len(raw_seqs) > max_seqs:
            clades[filename] = random.sample(raw_seqs, max_seqs)
        else:
            clades[filename] = raw_seqs

    # 4. Create the new expanded DataFrame filled with NaNs
    df_new = pd.DataFrame(np.nan, index=all_labels, columns=all_labels)
    
    # Copy the already computed distances into the new grid
    df_new.loc[valid_existing, valid_existing] = df_existing.loc[valid_existing, valid_existing]

    # Setup Aligner
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1

    # 5. Calculate Diagonal (Intra-species) only for NEW files
    for label in new_labels:
        seqs = clades[label]
        if len(seqs) <= 1:
            df_new.loc[label, label] = 0.0
            print(f"[{label}] 1 sequence: Intra-species distance = 0.0")
        else:
            intra_distances = [calc_k2p_core(s1, s2, aligner) for s1, s2 in itertools.combinations(seqs, 2)]
            max_dist = np.nanmax(intra_distances)
            df_new.loc[label, label] = max_dist
            print(f"[{label}] {len(seqs)} sequences: Max Intra-species distance = {max_dist:.5f}")

    # 6. Calculate Off-Diagonal (Inter-species) only where cell is NaN
    for i, j in itertools.combinations(range(len(all_labels)), 2):
        label_A = all_labels[i]
        label_B = all_labels[j]
        
        # Skip if this pair was already copied from the existing matrix
        if pd.notna(df_new.loc[label_A, label_B]):
            continue
            
        print(f"\r\033[K- Computing new pair: {label_A} vs {label_B}", end="", flush=True)
        
        seqs_A = clades[label_A]
        seqs_B = clades[label_B]
        
        inter_distances = [calc_k2p_core(sA, sB, aligner) for sA, sB in itertools.product(seqs_A, seqs_B)]
        mean_dist = np.nanmean(inter_distances)
        
        df_new.loc[label_A, label_B] = mean_dist
        df_new.loc[label_B, label_A] = mean_dist

    # 7. Export updated DataFrame
    print("\nCalculations complete.")
    df_new.to_csv(matrix_csv)
    print(f"Updated matrix successfully saved to {matrix_csv}")

    return df_new


def plot_clustered_distance_matrix(csv_file_path: str, distance_name: str):
    """
    Loads a pairwise distance matrix and plots a clustered heatmap.
    """
    # 1. Load the distance matrix from CSV
    # Ensure the first column (accession labels) is used as the index
    df = pd.read_csv(csv_file_path, index_col=0)
    # --- CLEAN THE LABELS HERE ---
    # Replace '_complete.fasta' (or '_complete_fasta') with an empty string
    df.index = df.index.str.replace('_complete.fasta', '', regex=False)
    df.columns = df.columns.str.replace('_complete.fasta', '', regex=False)
    # -----------------------------

    # Optional: Fill any NaN values (e.g., from sequences too divergent for K2P) 
    # with a structurally maximum value to prevent the clustering algorithm from crashing.
    max_dist = df.max().max()
    df.fillna(max_dist * 1.5, inplace=True)
    
    # 1. Store a clean copy for clustering (diagonal = 0.0)
    clustering_df = df.copy()
    np.fill_diagonal(clustering_df.values, 0.0)
    
    # Handle NaN values for clustering
    max_dist = clustering_df.max().max()
    clustering_df.fillna(max_dist * 1.5, inplace=True)
    
    # 2. Generate the clustergrid using the zero-diagonal matrix
    condensed_dist = ssd.squareform(clustering_df.values)
    linkage_matrix = sch.linkage(condensed_dist, method='average')
    
    # 3. Plot the Clustermap
    # A sequential colormap ('rocket_r' or 'mako_r') works best for distances. 
    # The '_r' reverses it so that 0 (identical) is light, and high distances are dark.
    g = sns.clustermap(
        df,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        cmap="YlOrRd", #"rocket_r",
        figsize=(12, 12),
        linewidths=0.5,       # Adds gridlines between cells
        cbar_pos=(0.02, 0.8, 0.05, 0.18), # Moves the colorbar out of the way
        cbar_kws={'label': f'{distance_name} Distance'}
    )
    
    # 4. Formatting adjustments
    # Ensure the labels are rotated for readability
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, ha='right', fontsize=9)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.title(f"Clustered Heatmap of mtDNA {distance_name} Distances", fontsize=16, pad=20)
    
    # Save the high-resolution figure
    plt.savefig(f"{distance_name.lower()}_clustermap.png", dpi=300, bbox_inches='tight')
    plt.show()


def linkage_to_topology_newick(linkage_matrix, labels):
    """
    Converts a SciPy linkage matrix into a pure topology Newick string 
    (no branch lengths, strictly nested relationships).
    """
    n_samples = len(labels)
    nodes = {}
    
    for i, row in enumerate(linkage_matrix):
        left_idx = int(row[0])
        right_idx = int(row[1])
        
        # Format left child
        left_str = labels[left_idx] if left_idx < n_samples else nodes[left_idx]
            
        # Format right child
        right_str = labels[right_idx] if right_idx < n_samples else nodes[right_idx]
            
        # Combine into parent node without any distance annotations
        current_node_idx = n_samples + i
        nodes[current_node_idx] = f"({left_str},{right_str})"
        
    root_idx = n_samples + len(linkage_matrix) - 1
    return "(" + nodes[root_idx] + ");"


def plot_uniform_cladogram(csv_file_path: str, graph_on=True):
    """
    Plots a clean cladogram where all branch lengths are forced to be 
    homogeneous, displaying pure topological relationships.
    """
    df = pd.read_csv(csv_file_path, index_col=0)
    labels = df.index.str.replace('_complete.fasta', '', regex=False).tolist()
    
    clustering_df = df.copy()
    np.fill_diagonal(clustering_df.values, 0.0)
    max_dist = clustering_df.max().max()
    clustering_df.fillna(max_dist * 1.5, inplace=True)
    
    # Generate standard linkage matrix
    condensed_dist = ssd.squareform(clustering_df.values)
    linkage_matrix = sch.linkage(condensed_dist, method='average')
    
    # --- CLADOGRAM HACK ---
    # Replace the actual distance heights (column 2) with sequential step ranks (0, 1, 2, ...)
    # This forces every branch merge to occupy an equal visual unit interval.
    linkage_matrix[:, 2] = np.arange(len(linkage_matrix))
    # ----------------------
    
    if graph_on:
        plt.figure(figsize=(10, 14))
        
        dendro = sch.dendrogram(
            linkage_matrix,
            labels=labels,
            orientation='left',
            leaf_font_size=11,
            color_threshold=0  # Sets all branches to a uniform color
        )
        
        plt.title("Phylogenetic Cladogram of Felidae (Uniform Topology)", fontsize=13, pad=15)
        plt.xlabel("Clustering Steps (Topology Only)", fontsize=11)
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # Hide the x-axis tick values since they are now arbitrary ranks
        ax.set_xticks([])
        
        plt.tight_layout()
        plt.show()

    topology_newick = linkage_to_topology_newick(linkage_matrix, labels)
    return topology_newick
    
    
def tuple_to_newick(node):
    """
    Recursively converts a nested tuple/string tree structure into a Newick string.
    """
    # If the node is a string, it's a leaf node (species name)
    if isinstance(node, str):
        return node
    
    # Otherwise, it's a tuple of child nodes/sub-trees
    # Recursively convert each child and join them with commas inside parentheses
    children_str = ",".join(tuple_to_newick(child) for child in node)
    return f"({children_str})"


def plot_newick_cladogram(newick_str: str, title: str = "Phylogenetic Cladogram (Uniform Topology)"):
    """
    Parses a Newick string and plots a pure topology cladogram 
    with uniform branch lengths on a Matplotlib axis.
    """
    # 1. Parse the Newick string
    tree = Phylo.read(StringIO(newick_str), "newick")
    
    # 2. Strip all branch lengths to force homogeneous spacing
    for clade in tree.find_clades():
        clade.branch_length = None
        
    # 3. Setup the Matplotlib canvas
    fig, ax = plt.subplots(figsize=(10, 14))
    
    # 4. Draw the tree onto the axis
    # By default, Phylo.draw adds a box outline, we will clean it below
    Phylo.draw(tree, axes=ax, do_show=False)
    
    # 5. Format to match your clean aesthetic
    plt.title(title, fontsize=13, pad=15)
    plt.xlabel("Branching Events (Topology Only)", fontsize=11)
    plt.ylabel("")  # Clear the default 'Taxa' label if desired
    
    # Clean the spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Hide axis values
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()


def plot_justified_newick_cladogram(newick_str: str, title: str = "Phylogenetic Cladogram (Uniform Topology)", y=14):
    # 1. Parse the Newick string
    tree = Phylo.read(StringIO(newick_str), "newick")
    
    # 2. Force all branches to a uniform length of 1.0 and set color to Matplotlib blue
    for clade in tree.find_clades():
        clade.branch_length = 1.0
        clade.color = '#1f77b4' 
        
    # 3. RIGHT-JUSTIFICATION MATH
    # Calculate the total distance from the root to every node
    depths = tree.depths()
    max_depth = max(depths.values())
    
    # Pad only the terminal branches (the leaves) so they all stretch to max_depth
    for leaf in tree.get_terminals():
        padding_needed = max_depth - depths[leaf]
        leaf.branch_length += padding_needed
        
    # 4. Setup the Matplotlib canvas
    fig, ax = plt.subplots(figsize=(10, y))
    
    # 5. Draw the tree onto the axis
    Phylo.draw(tree, axes=ax, do_show=False)
    
    # 6. Format to match your clean aesthetic
    plt.title(title, fontsize=13, pad=15)
    plt.xlabel("Branching Events (Topology Only)", fontsize=11)
    plt.ylabel("")  
    
    # Clean the spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    

##########################
# Rank-frequency analysis
##########################


def get_global_frequencies(folder_path: str, detail: bool = False) -> dict:
    """
    Overall frequency distribution across all sequences in all FASTA files.
    The fractions are of ALL characters, so ambiguity codes are included in the
    denominator. Returns 'A', 'C', 'G', 'T' and 'non-ACGT' (summing to 1).
    With detail=True, every non-ACGT letter (N, R, Y, ...) is listed on its own
    instead of the single 'non-ACGT' entry.
    """
    global_counts = Counter()
    total_bases = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(('.fasta', '.fa', '.fas')):
            filepath = os.path.join(folder_path, filename)
            print(filepath)
            for record in SeqIO.parse(filepath, "fasta"):
                seq = str(record.seq).upper()
                global_counts.update(seq)
                total_bases += len(seq)

    if total_bases == 0:
        return {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0, 'non-ACGT': 0.0}

    freqs = {nuc: global_counts.get(nuc, 0) / total_bases for nuc in 'ACGT'}
    others = {ch: n for ch, n in global_counts.items() if ch not in 'ACGT'}
    if detail:
        freqs.update({ch: n / total_bases for ch, n in sorted(others.items())})
    else:
        freqs['non-ACGT'] = sum(others.values()) / total_bases
    return freqs



def tokenize_by_space_with_empty(document, space_char='A', empty_token='O'):
    """Returns a LIST of tokens, preventing premature string iteration."""
    raw_splits = document.upper().split(space_char.upper())
    
    # Return the list directly. Do not use " ".join() here.
    return [empty_token if w == '' else w for w in raw_splits]
    

def get_tokenized_documents(fasta_path, space_char='A', words_per_chunk=150, step_size=75, max_seqs_per_file=3):
    """
    Slices a genome into overlapping documents based on whole word counts,
    preventing any biological sequences from being arbitrarily fragmented.
    """
    documents = []
    
    # Load all records to evaluate the total count for this specific species/file
    records = list(SeqIO.parse(fasta_path, "fasta"))
    
    # Subsample if the file contains more sequences than the allowed maximum
    if len(records) > max_seqs_per_file:
        random.seed(42) # Ensures the same genomes are selected across different runs
        records = random.sample(records, max_seqs_per_file)
        
    for record in records:
        seq = str(record.seq)
    
        # 1. Parse the entire genome into an ordered list of variable-length words
        all_words = tokenize_by_space_with_empty(seq, space_char)
        
        # 2. Slide a window over the LIST of words, not the raw nucleotide string
        #for i in range(0, len(all_words) - words_per_chunk + 1, step_size):
        for i in range(0, len(all_words), step_size):
            
            # Extract the exact block of whole words
            chunk_of_words = all_words[i : i + words_per_chunk]
            
            # Join them with standard spaces so TfidfVectorizer can process them natively
            documents.append(" ".join(chunk_of_words))
        
    return documents
    

def build_hybrid_lsa_pipeline_species(folder_path: str, space_char='A', words_per_chunk=150, step_size=75, max_seqs_per_file=3, n_components=25):
    corpus = []
    doc_labels = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.fasta', '.fa', '.fas')):
            filepath = os.path.join(folder_path, filename)
            
            # 1. Clean the label immediately so you don't have to do it later
            label = os.path.splitext(filename)[0].replace('_complete', '')
            
            docs = get_tokenized_documents(filepath, space_char, words_per_chunk, step_size, max_seqs_per_file)
            
            # 2. THE CRITICAL FIX: Concatenate all chunks into a single "document" for this species.
            # Now, TF-IDF and SVD will learn how species differ from each other, 
            # rather than how genes differ from each other.
            species_document = " ".join(docs)
            
            corpus.append(species_document)
            doc_labels.append(label)
            
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
        # 3. LOWER MIN_DF: Since the corpus is ~40 species, 
        # min_df=10 would drop motifs unless they appear in 25% of all species. 
        # min_df=2 ensures we keep motifs shared by at least sister taxa.
        min_df=3, 
        ngram_range=(1, 2),
        #max_df=0.5,
        binary=True
    )
    
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # 4. NUMBER OF SVD COMPONENTS: 3 is too small to capture 40 species. 
    # 25 gives the vector space enough room to separate the lineages. 
    # (Note: n_components must be strictly less than the number of species).
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_matrix = svd.fit_transform(tfidf_matrix)
    
    stable_vocabulary = vectorizer.get_feature_names_out()
    
    return lsa_matrix, doc_labels, svd, stable_vocabulary, tfidf_matrix


def build_hybrid_rf_pipeline_species(folder_path: str, space_char='A', max_seqs_per_file=3, max_features=500):
    corpus = []
    doc_labels = []
    
    # Force the sliding window to encompass the entire mtDNA genome
    words_per_chunk = 1000
    step_size = 1000
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.fasta', '.fa', '.fas')):
            filepath = os.path.join(folder_path, filename)
            label = os.path.splitext(filename)[0].replace('_complete', '')
            
            # Retrieves whole sequences, restricted by max_seqs_per_file
            docs = get_tokenized_documents(filepath, space_char, words_per_chunk, step_size, max_seqs_per_file)
            
            # Join the whole sequences into a single species document
            species_document = " ".join(docs)
            
            corpus.append(species_document)
            doc_labels.append(label)
            
    vectorizer = CountVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
        max_features=max_features, 
        ngram_range=(1, 1),
        binary=True
    )
    
    raw_counts = vectorizer.fit_transform(corpus)
    relative_frequencies = normalize(raw_counts, norm='l1', axis=1)
    stable_vocabulary = vectorizer.get_feature_names_out()
    
    return relative_frequencies, doc_labels, stable_vocabulary
    

def build_population_euclidean_matrix(family: str, directory_path: str, file_extension: str = "*.fasta", max_seqs: int = 2, seed: int = 42, max_features: int = 5000, space_char: str = 'A'):
    """Generates an inter/intra-species Euclidean distance matrix using binary rank-frequency NLP."""
    
    # Lock the random seed so subsampling yields the exact same sequences on repeat runs
    random.seed(seed)
    
    search_pattern = os.path.join(directory_path, file_extension)
    fasta_files = sorted(glob.glob(search_pattern))
    
    if not fasta_files:
        print("No FASTA files found.")
        return None

    clades = {}
    file_labels = []
    
    # --- 1. SUBSAMPLING (Exactly as you structured it) ---
    for file in fasta_files:
        filename = os.path.basename(file)
        file_labels.append(filename)
        
        records = list(SeqIO.parse(file, "fasta"))
        raw_seqs = [str(rec.seq).upper() for rec in records]
        
        # Check and apply the random subsampling limit
        if len(raw_seqs) > max_seqs:
            print(f"[{filename}] Found {len(raw_seqs)} sequences. Randomly subsampling to {max_seqs}.")
            clades[filename] = random.sample(raw_seqs, max_seqs)
        else:
            clades[filename] = raw_seqs
            
    n = len(file_labels)
    dist_matrix = np.zeros((n, n))
    print(f"Loaded {n} files.")

    # --- 2. THE NLP VECTORIZATION PHASE ---
    print("Vectorizing the subsampled population...")
    corpus = []
    seq_mapping = {} # Maps (filename, local_seq_index) to the global corpus index
    global_idx = 0
    
    for label, seqs in clades.items():
        for i, seq in enumerate(seqs):
            # Tokenize by the chosen character
            words = tokenize_by_space_with_empty(seq, space_char)
            corpus.append(" ".join(words))
            seq_mapping[(label, i)] = global_idx
            global_idx += 1
            
    # Apply the exact math that scored 36/82
    vectorizer = CountVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
        max_features=max_features, 
        ngram_range=(1, 1),
        binary=True 
    )
    
    # Convert to dense array for SciPy euclidean function
    raw_counts = vectorizer.fit_transform(corpus)
    relative_freqs = normalize(raw_counts, norm='l1', axis=1).toarray()

    # --- 3. CALCULATE DIAGONAL (Intra-species Maximum) ---
    for i in range(n):
        label = file_labels[i]
        seqs = clades[label]
        
        if len(seqs) <= 1:
            dist_matrix[i, i] = 0.0
            print(f"[{label}] 1 sequence: Intra-species distance = 0.0")
        else:
            intra_distances = []
            # Iterate by index to fetch the pre-calculated Euclidean vectors
            for s1_idx, s2_idx in itertools.combinations(range(len(seqs)), 2):
                vec1 = relative_freqs[seq_mapping[(label, s1_idx)]]
                vec2 = relative_freqs[seq_mapping[(label, s2_idx)]]
                intra_distances.append(euclidean(vec1, vec2))
            
            max_dist = np.nanmax(intra_distances)
            dist_matrix[i, i] = max_dist
            print(f"[{label}] {len(seqs)} sequences: Max Intra-species distance = {max_dist:.5f}")

    # --- 4. CALCULATE OFF-DIAGONAL (Inter-species Average) ---
    for i, j in itertools.combinations(range(n), 2):
        print(f'\ri, j = {i} {j}', end='', flush=True)
        label_A = file_labels[i]
        label_B = file_labels[j]
        
        seqs_A = clades[label_A]
        seqs_B = clades[label_B]
        
        inter_distances = []
        for sA_idx, sB_idx in itertools.product(range(len(seqs_A)), range(len(seqs_B))):
            vecA = relative_freqs[seq_mapping[(label_A, sA_idx)]]
            vecB = relative_freqs[seq_mapping[(label_B, sB_idx)]]
            inter_distances.append(euclidean(vecA, vecB))
            
        mean_dist = np.nanmean(inter_distances)
        
        # Mirror the matrix
        dist_matrix[i, j] = mean_dist
        dist_matrix[j, i] = mean_dist
        
    # --- 5. EXPORT ---
    df = pd.DataFrame(dist_matrix, index=file_labels, columns=file_labels)
    df.to_csv(f"{family}_euclidean_matrix.csv")
    print("\nMatrix successfully saved to population_euclidean_matrix.csv")
    
    return df
    
    
    
def species_tfidf(folder, space_char='A', words_per_chunk=150, step_size=75,
                  max_seqs_per_file=3, **vec):
    """Species-level TF-IDF matrix with the vectorizer settings open.
    Defaults are those of build_hybrid_lsa_pipeline_species in mtdna.py.
    Extra keywords go to TfidfVectorizer, e.g. min_df=3, ngram_range=(1, 3), max_df=0.9.
    words_per_chunk=None: whole genomes, no windows."""
    params = dict(token_pattern=r"(?u)\b\w+\b", lowercase=False,
                  min_df=3, ngram_range=(1, 2), binary=True)
    params.update(vec)
    corpus, labels = [], []
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(('.fasta', '.fa', '.fas')):
            continue
        path = os.path.join(folder, fn)
        if words_per_chunk is None:
            records = list(SeqIO.parse(path, "fasta"))
            if len(records) > max_seqs_per_file:
                random.seed(42)                      # same selection as in mtdna.py
                records = random.sample(records, max_seqs_per_file)
            docs = [" ".join(tokenize_by_space_with_empty(str(r.seq), space_char)) for r in records]
        else:
            docs = get_tokenized_documents(path, space_char, words_per_chunk, step_size, max_seqs_per_file)
        corpus.append(" ".join(docs))
        labels.append(os.path.splitext(fn)[0].replace('_complete', ''))
    vectorizer = TfidfVectorizer(**params)
    X = vectorizer.fit_transform(corpus)
    return X, labels, vectorizer

    
Panthera_lineage = (
    (
        ("Panthera_tigris", "Panthera_uncia"),
        ("Panthera_onca",
         ("Panthera_leo", "Panthera_pardus"),
        )
    ),
    ("Neofelis_nebulosa", "Neofelis_diardi")    
)

BayCat_lineage = (("Catopuma_temminckii", "Catopuma_badia"), "Pardofelis_marmorata")
Caracal_lineage = (("Caracal_caracal", "Caracal_aurata"), "Leptailurus_serval")

_wiedii_jacobita = ("Leopardus_wiedii", "Leopardus_jacobita")
_pampas_cats = ("Leopardus_braccatus", "Leopardus_colocola", "Leopardus_garleppi",
                "Leopardus_fasciatus", "Leopardus_pajeros")
_ocelot_proper = (("Leopardus_pardalis", _wiedii_jacobita), _pampas_cats)
_tigercat_proper = ("Leopardus_tilcayo", ("Leopardus_pardinoides", "Leopardus_tigrinus"))
_oncifelis_core = (("Leopardus_guttulus", "Leopardus_emiliae"), ("Leopardus_geoffroyi", "Leopardus_guigna"))
Ocelot_lineage = (_ocelot_proper, (_oncifelis_core, _tigercat_proper))

Lynx_lineage = ("Lynx_rufus", ("Lynx_canadensis", ("Lynx_lynx", "Lynx_pardinus")))
Puma_lineage = ("Acinonyx_jubatus", ("Puma_concolor", "Herpailurus_yagouaroundi"))

LeopardCat_lineage = ("Otocolobus_manul", 
                      ("Prionailurus_rubiginosus", 
                       ("Prionailurus_planiceps", 
                        ("Prionailurus_viverrinus", 
                         ("Prionailurus_bengalensis", "Prionailurus_javanensis")
                        )
                       )
                      )
                     )

DomesticCat_lineage = ("Felis_chaus", 
                       ("Felis_nigripes", 
                        ("Felis_margarita", 
                         ("Felis_silvestris", 
                          ("Felis_bieti", 
                           ("Felis_lybica", "Felis_catus")
                          )
                         )
                        )
                       )
                      )


FELIDAE_REFERENCE = (
    Panthera_lineage,
    (
        Caracal_lineage,
        (
            Ocelot_lineage,
            (
                (BayCat_lineage, Lynx_lineage),
                (Puma_lineage, (LeopardCat_lineage, DomesticCat_lineage)),
            ),
        ),
    ),
)


CANIDAE_REFERENCE = (
  (
    'Urocyon_cinereoargenteus',
    'Urocyon_littoralis'
  ),
  (
    (
      'Otocyon_megalotis',
      (
        (
          'Nyctereutes_procyonoides',
          'Nyctereutes_viverrinus'
        ),
        (
          (
            'Vulpes_zerda',
            'Vulpes_cana'
          ),
          (
            'Vulpes_chama',
            (
              (
                (
                  (
                    'Vulpes_vulpes',
                    (
                      'Vulpes_rueppellii',
                      'Vulpes_pallida'
                    )
                  ),
                  'Vulpes_bengalensis'
                ),
                (
                  'Vulpes_corsac',
                  'Vulpes_ferrilata'
                )
              ),
              (
                'Vulpes_lagopus',
                (
                  'Vulpes_macrotis',
                  'Vulpes_velox'
                )
              )
            )
          )
        )
      )
    ),
    (
      (
        (
          'Speothos_venaticus',
          'Chrysocyon_brachyurus'
        ),
        (
          'Atelocynus_microtis',
          (
            'Cerdocyon_thous',
            (
              'Lycalopex_vetula',
              (
                'Lycalopex_sechurae',
                (
                  (
                    'Lycalopex_fulvipes',
                    'Lycalopex_gymnocerca'
                  ),
                  (
                    'Lycalopex_grisea',
                    'Lycalopex_culpaeus'
                  )
                )
              )
            )
          )
        )
      ),
      (
        (
          'Lupulella_adusta',
          'Lupulella_mesomelas'
        ),
        (
          'Lycaon_pictus',
          (
            'Cuon_alpinus',
            (
              'Canis_aureus',
              (
                'Canis_simensis',
                (
                  'Canis_lupaster',
                  (
                    (
                      'Canis_latrans',
                      (
                        'Canis_rufus',
                        'Canis_lycaon'
                      )
                    ),
                    (
                      'Canis_lupus',
                      'Canis_familiaris'
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
  )
)