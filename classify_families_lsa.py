import os
import numpy as np

from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import f1_score

from sklearn.model_selection import GroupKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC

# Import the custom tokenization function from mtdna.py
from mtdna import get_tokenized_documents

def load_family_corpus(directory_path, label_name, words_per_chunk=150, step_size=75, max_seqs_per_file=3):
    """
    Returns corpus chunks, labels, AND groups (the source filename) to prevent data leakage.
    """
    corpus = []
    labels = []
    groups = []
    
    if not os.path.exists(directory_path):
        print(f"Warning: Directory '{directory_path}' not found.")
        return corpus, labels, groups

    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(('.fasta', '.fa', '.fas')):
            filepath = os.path.join(directory_path, filename)
            
            docs = get_tokenized_documents(
                filepath, 
                space_char='A', 
                words_per_chunk=words_per_chunk, 
                step_size=step_size, 
                max_seqs_per_file=max_seqs_per_file
            )
            
            corpus.extend(docs)
            labels.extend([label_name] * len(docs))
            groups.extend([filename] * len(docs)) # Tag every chunk with its source file
            
    return corpus, labels, groups


def group_permutation_test(pipeline, X, y, groups, cv, n_perm=200, seed=42, n_jobs=-1):
    """Permute the family label among species (files), repeat the CV, return
    (observed, null_scores, p_value). Statistic: mean macro-F1 over the folds.
    TF-IDF and SVD never see the labels, so they are fitted once per fold and
    only the SVM is refitted for each permutation."""
    y = np.asarray(y)
    groups = np.asarray(groups)

    folds = []
    for train_idx, test_idx in cv.split(X, y, groups):
        tfidf = clone(pipeline.named_steps['tfidf'])
        svd = clone(pipeline.named_steps['svd'])
        Ztr = svd.fit_transform(tfidf.fit_transform([X[i] for i in train_idx]))
        Zte = svd.transform(tfidf.transform([X[i] for i in test_idx]))
        folds.append((train_idx, test_idx, Ztr, Zte))

    def mean_f1(labels):
        scores = []
        for train_idx, test_idx, Ztr, Zte in folds:
            clf = clone(pipeline.named_steps['clf'])
            clf.fit(Ztr, labels[train_idx])
            scores.append(f1_score(labels[test_idx], clf.predict(Zte), average='macro'))
        return float(np.mean(scores))

    observed = mean_f1(y)                      # true labels, same statistic

    species = np.unique(groups)
    species_label = np.array([y[groups == s][0] for s in species])
    pos = {s: i for i, s in enumerate(species)}
    group_pos = np.array([pos[g] for g in groups])

    rng = np.random.default_rng(seed)
    perms = [rng.permutation(species_label) for _ in range(n_perm)]
    null = np.array(Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(mean_f1)(p[group_pos]) for p in perms))

    p_value = (1 + np.sum(null >= observed)) / (1 + n_perm)
    return observed, null, p_value


def main():
    felidae_dir = 'felidae_dedup'
    canidae_dir = 'canidae_dedup'

    scoring = "f1_macro"
    
    print("Loading and tokenizing corpora...")
    
    fel_corpus, fel_labels, fel_groups = load_family_corpus(felidae_dir, label_name=0)
    print(f"Loaded {len(fel_corpus)} document chunks for Felidae.")
    
    can_corpus, can_labels, can_groups = load_family_corpus(canidae_dir, label_name=1)
    print(f"Loaded {len(can_corpus)} document chunks for Canidae.")
    
    X = fel_corpus + can_corpus
    y = fel_labels + can_labels
    groups = fel_groups + can_groups
    
    if not X:
        print("Error: No data loaded.")
        return

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            lowercase=False,
            min_df=3,
            ngram_range=(1, 2),
            binary=True 
        )),
        ('svd', TruncatedSVD(n_components=66, random_state=42)),
        ('clf', SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42))
    ])
    
    # Switch to GroupKFold to ensure no chunks from the same species cross the train/test barrier
    cv = GroupKFold(n_splits=5)
    
    print("\nRunning 5-fold strict species-level cross-validation...")
    # Pass the groups parameter to the cross_val_score function
    cv_scores = cross_val_score(pipeline, X, y, groups=groups, cv=cv, scoring=scoring)
    
    print(f"Cross-Validation Accuracies: {[round(score, 3) for score in cv_scores]}")
    print(f"Mean Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # Replace your cross_val_score line with this:
    cv_results = cross_validate(pipeline, X, y, groups=groups, cv=cv, 
                                scoring=scoring, return_train_score=True)

    print(f"Mean Train Accuracy: {cv_results['train_score'].mean():.4f}")
    print(f"Mean Test Accuracy:  {cv_results['test_score'].mean():.4f}")


    print("\nRunning group-level permutation test...")
    observed, null, p = group_permutation_test(pipeline, X, y, groups, cv, n_perm=1000)
    np.save("null_scores_linear.npy", null)     # change the name for RBF
    print(f"Observed: {observed:.4f}")
    print(f"Null: mean {null.mean():.3f}, SD {null.std(ddof=1):.3f}, "
          f"95th percentile {np.percentile(null, 95):.3f}, max {null.max():.3f}")
    print(f"Permutations: {len(null)}, empirical p = {p:.4f}")


if __name__ == "__main__":
    main()