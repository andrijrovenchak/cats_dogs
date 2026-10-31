from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.spatial.distance import squareform, pdist, euclidean, cosine
from scipy.cluster.hierarchy import linkage

from ete3 import Tree

from mtdna import (
    get_global_frequencies,
    build_population_jaccard_matrix, build_population_k2p_matrix, build_population_mash_matrix,
    update_population_k2p_matrix,
    plot_clustered_distance_matrix, plot_uniform_cladogram,
    plot_justified_newick_cladogram,
    tuple_to_newick, linkage_to_topology_newick,
    build_population_euclidean_matrix, 
    build_hybrid_rf_pipeline_species,
    build_hybrid_lsa_pipeline_species,
    FELIDAE_REFERENCE, CANIDAE_REFERENCE
)

from pretty_newick import convert


print("Character frequencies in the FASTA folders:")
for family in ['felidae', 'canidae']:
    print(family.upper())
    freqs = get_global_frequencies(f'{family}_dedup')
    print(freqs)


XLS_BUILT = True
if not XLS_BUILT:
    # CATS & DOGS matrices generation
    for family in ['felidae', 'canidae']:
        matrix = build_population_euclidean_matrix(family, f'{family}_dedup', max_seqs=3,  max_features=5000)
        matrix = build_population_k2p_matrix(family, f"{family}_dedup",  max_seqs=3)

    for kmers in range(3, 22):
        print("k-mers:", kmers)
        for family in ['felidae', 'canidae']:
            print(family.upper())
            matrix = build_population_jaccard_matrix(family, f"{family}_dedup",  max_seqs=3, k=kmers)
            matrix = build_population_mash_matrix(family, f"{family}_dedup",  max_seqs=3, k=kmers)

kmers=13
### CATS, DOGS ###
for family, fam_ref in zip(['felidae', 'canidae'], [FELIDAE_REFERENCE, CANIDAE_REFERENCE]):
    print(family.upper())
    newick_reference = tuple_to_newick(fam_ref) + ";"
    t0 = Tree(newick_reference)
    
    newick_jaccard = plot_uniform_cladogram(f'{family}_jaccard_matrix-k{kmers}.csv', graph_on=False)
    newick_mash = plot_uniform_cladogram(f'{family}_mash_matrix-k{kmers}.csv', graph_on=False)
    newick_k2p = plot_uniform_cladogram(f'{family}_k2p_matrix.csv', graph_on=False)
    newick_euclidean = plot_uniform_cladogram(f'{family}_euclidean_matrix.csv', graph_on=False)

    trees = {
        'jaccard':   newick_jaccard,
        'mash':      newick_mash,
        'k2p':       newick_k2p,
        'euclidean': newick_euclidean,
    }
    for name, nw in trees.items():
        # Robinson-Foulds distance
        rf, max_rf, common_leaves, _, _, _, _ = Tree(nw).robinson_foulds(t0)
        print(f"   {name.capitalize() if name != 'k2p' else name.upper():10}: {rf} / {max_rf} ({rf / max_rf:.3f})")
        n = len(Tree(nw, format=1))
        open(f'{family}{n if family == "canidae" else ""}_{name}_tree.nwk', 'w').write(convert(nw))
    
    #species_lsa_matrix, species_labels, svd, stable_vocabulary, tfidf_matrix = build_hybrid_lsa_pipeline_species(
    #    f'{family}_dedup', words_per_chunk=100, step_size=100)
    #print("Aggregated shape:", species_lsa_matrix.shape, tfidf_matrix.shape)

    # 1. Run the exact pipeline
    lsa_matrix, doc_labels, svd, stable_vocabulary, tfidf_matrix = build_hybrid_lsa_pipeline_species(f'{family}_dedup')
    print("   LSA Aggregated shape:", lsa_matrix.shape, tfidf_matrix.shape)

    # 2. Find unique labels and keep only the first occurrence of each row
    unique_labels = []
    unique_indices = []
    for i, label in enumerate(doc_labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_indices.append(i)

    # 3. Filter the TF-IDF matrix to keep only unique species
    unique_tfidf_matrix = tfidf_matrix[unique_indices]

    # 4. Calculate distances directly on the high-dimensional TF-IDF matrix
    tfidf_dist = cosine_distances(unique_tfidf_matrix)
    condensed_tfidf = squareform(tfidf_dist)

    # 5. Build the tree
    linkage_tfidf = linkage(condensed_tfidf, method='average')
    newick_tfidf = linkage_to_topology_newick(linkage_tfidf, unique_labels)

    # 6. Load both trees
    t_tfidf = Tree(newick_tfidf)
    t_ref = Tree(newick_reference)

    # 7. Ensure both trees only compare the exact same shared species
    t_ref.prune(t_tfidf.get_leaf_names(), preserve_branch_length=False)

    # 8. Calculate Robinson-Foulds
    rf_tfidf, max_rf, _, _, _, _, _ = t_tfidf.robinson_foulds(t_ref)
    print(f"   {'TF-IDF':10}: {rf_tfidf} / {max_rf} ({rf_tfidf / max_rf:.3f})")

    rf_matrix, doc_labels, vocab = build_hybrid_rf_pipeline_species(f'{family}_dedup', max_seqs_per_file=3, max_features=5000)

    # Deduplicate to prevent TreeError
    unique_labels = []
    unique_indices = []
    for i, label in enumerate(doc_labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_indices.append(i)

    unique_rf_matrix = rf_matrix[unique_indices]

    # Convert the sparse matrix to a dense NumPy array for pdist
    dense_rf_matrix = unique_rf_matrix.toarray()

    # Test 1: Cosine Distance 
    condensed_cosine = pdist(dense_rf_matrix, metric='cosine')
    linkage_cosine = linkage(condensed_cosine, method='average')
    newick_cosine = linkage_to_topology_newick(linkage_cosine, unique_labels)

    # Test 2: Euclidean Distance 
    condensed_euclidean = pdist(dense_rf_matrix, metric='euclidean')
    linkage_euclidean = linkage(condensed_euclidean, method='average')
    newick_euclidean = linkage_to_topology_newick(linkage_euclidean, unique_labels)

    # Load the trees to check the Robinson-Foulds scores
    t_cosine = Tree(newick_cosine)
    t_euclidean = Tree(newick_euclidean)

    # (Assuming t_ref is already loaded and pruned to unique_labels)
    rf_cosine, max_rf, _, _, _, _, _ = t_cosine.robinson_foulds(t_ref)
    rf_euclidean, _, _, _, _, _, _ = t_euclidean.robinson_foulds(t_ref)

    print(f"   Cosine(sp): {rf_cosine} / {max_rf} ({rf_cosine / max_rf:.3f})")
    print(f"   Euclid(sp): {rf_euclidean} / {max_rf} ({rf_euclidean / max_rf:.3f})")

    n = len(Tree(newick_tfidf, format=1))
    open(f'{family}{n if family == "canidae" else ""}_tfidf_tree.nwk', 'w').write(convert(newick_tfidf))
    open(f'{family}{n if family == "canidae" else ""}_cosine-species_tree.nwk', 'w').write(convert(newick_cosine))
    open(f'{family}{n if family == "canidae" else ""}_euclidean-species_tree.nwk', 'w').write(convert(newick_euclidean))

    print('=' * 80)
    print()