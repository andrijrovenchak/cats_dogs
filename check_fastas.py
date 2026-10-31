import os
from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC

# Import your existing load_family_corpus and get_tokenized_documents functions
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


def predict_outgroup(fasta_path, pipeline, chunk_params):
    """
    Tokenizes an unseen FASTA, predicts the class for every chunk, 
    and returns the majority vote.
    """
    if not os.path.exists(fasta_path):
        print(f"Error: {fasta_path} not found.")
        return

    # 1. Tokenize the unseen file using the exact same parameters as training
    docs = get_tokenized_documents(
        fasta_path, 
        space_char='A', 
        words_per_chunk=chunk_params['words_per_chunk'], 
        step_size=chunk_params['step_size'], 
        max_seqs_per_file=1  # We only need to read the single outgroup genome
    )
    
    if not docs:
        print(f"Failed to extract chunks from {fasta_path}.")
        return

    # 2. Predict every chunk
    predictions = pipeline.predict(docs)
    
    # 3. Tally the votes
    counts = Counter(predictions)
    total_chunks = len(predictions)
    feliformia_votes = counts.get(0, 0)
    caniformia_votes = counts.get(1, 0)
    
    fel_pct = (feliformia_votes / total_chunks) * 100
    can_pct = (caniformia_votes / total_chunks) * 100
    
    winner = "Feliformia (Cat-like)" if feliformia_votes > caniformia_votes else "Caniformia (Dog-like)"
    
    print(f"--- Results for {fasta_path} ---")
    print(f"Total Chunks Evaluated: {total_chunks}")
    print(f"Class 0 (Felidae-like) Votes: {feliformia_votes} ({fel_pct:.1f}%)")
    print(f"Class 1 (Canidae-like) Votes: {caniformia_votes} ({can_pct:.1f}%)")
    print(f"Final Classification: {winner}\n")

def main():
    # Set the parameters you used for your successful stress test
    chunk_params = {'words_per_chunk': 150, 'step_size': 75}
    
    print("Loading training corpora...")
    fel_corpus, fel_labels, _ = load_family_corpus('felidae_dedup', 0, **chunk_params)
    can_corpus, can_labels, _ = load_family_corpus('canidae_dedup', 1, **chunk_params)
    
    X_train = fel_corpus + can_corpus
    y_train = fel_labels + can_labels
    
    # Build your proven LSA Pipeline
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

    print("Training pipeline on all available family data...")
    # Fit the pipeline on the ENTIRE dataset (no CV splits needed for external prediction)
    pipeline.fit(X_train, y_train)
    print("Training complete.\n")
    
    # Test the outgroups
    predict_outgroup('hyena.fasta', pipeline, chunk_params)
    predict_outgroup('bear.fasta', pipeline, chunk_params)

if __name__ == "__main__":
    main()