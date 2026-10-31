import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, cross_val_score, cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Import your custom tokenization function from mtdna.py
from mtdna import get_tokenized_documents

def load_genus_corpus(directory_path, target_genus, label_name, words_per_chunk=150, step_size=75, max_seqs_per_file=3):
    """
    Returns corpus chunks, labels, and groups for a specific genus within a family directory.
    """
    corpus = []
    labels = []
    groups = []
    
    if not os.path.exists(directory_path):
        print(f"Warning: Directory '{directory_path}' not found.")
        return corpus, labels, groups

    for filename in sorted(os.listdir(directory_path)):
        # Only process files that belong to the target genus
        if filename.startswith(target_genus) and filename.endswith(('.fasta', '.fa', '.fas')):
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

def main():
    canidae_dir = 'canidae_dedup'
    
    print("Loading and tokenizing corpora...")
    
    # Load Vulpes (Class 0)
    vulpes_corpus, vulpes_labels, vulpes_groups = load_genus_corpus(canidae_dir, target_genus='Vulpes', label_name=0)
    print(f"Loaded {len(vulpes_corpus)} document chunks for Vulpes.")
    
    # Load Canis (Class 1)
    canis_corpus, canis_labels, canis_groups = load_genus_corpus(canidae_dir, target_genus='Canis', label_name=1)
    print(f"Loaded {len(canis_corpus)} document chunks for Canis.")
    
    X = vulpes_corpus + canis_corpus
    y = vulpes_labels + canis_labels
    groups = vulpes_groups + canis_groups
    
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
        ('clf', SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42))
    ])
    
    cv = GroupKFold(n_splits=5)
    
    print("\nRunning 5-fold strict species-level cross-validation...")

    y = np.array(y); groups = np.array(groups)
    pred = cross_val_predict(pipeline, X, y, groups=groups, cv=cv)
    print(f"Accuracy: {accuracy_score(y, pred):.4f}")
    print(f"Macro-F1: {f1_score(y, pred, average='macro'):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y, pred):.4f}")
    print("Confusion matrix (rows: true Vulpes, Canis; columns: predicted):")
    print(confusion_matrix(y, pred))
    for g in sorted(set(groups)):
        m = groups == g
        if (pred[m] != y[m]).any():
            print(f"  {g}: {(pred[m] != y[m]).sum()} of {m.sum()} chunks wrong")


if __name__ == "__main__":
    main()