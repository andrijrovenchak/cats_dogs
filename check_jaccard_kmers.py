from ete3 import Tree

from mtdna import (
    plot_uniform_cladogram,
    tuple_to_newick,
    FELIDAE_REFERENCE, CANIDAE_REFERENCE
)


for kmers in range(3, 22):
    print(f"k-mers: {kmers:2d}", end=" ")

    for family, fam_ref in zip(['felidae', 'canidae'], [FELIDAE_REFERENCE, CANIDAE_REFERENCE]):
        newick_reference = tuple_to_newick(fam_ref) + ";"
        t0 = Tree(newick_reference)
        newick_jaccard = plot_uniform_cladogram(f'{family}_jaccard_matrix-k{kmers}.csv', graph_on=False)
        newick_mash = plot_uniform_cladogram(f'{family}_mash_matrix-k{kmers}.csv', graph_on=False)

        t_jacc = Tree(newick_jaccard)
        t_mash = Tree(newick_mash)

        rf_jacc, max_rf_jacc, _, _, _, _, _ = t_jacc.robinson_foulds(t0)
        rf_mash, max_rf_mash, _, _, _, _, _ = t_mash.robinson_foulds(t0)

        print(f"{family.capitalize()}: {rf_jacc} / {max_rf_jacc} ({rf_jacc / max_rf_jacc:.3f})", end=" | ")
        print(f"{rf_mash} / {max_rf_mash} ({rf_mash / max_rf_mash:.3f})", end=" ")
    print()