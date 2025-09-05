#!/usr/bin/env python3
import math
from collections import Counter
from typing import List, Tuple

import spacy
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessor import Preprocessor

# --- Setup ---
nlp = spacy.load("mk_core_news_lg")
ALLOWED_POS = {"NOUN", "PROPN", "ADJ", "VERB", "ADV"}

def lemmas_from_text(text: str) -> List[str]:
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop and token.pos_ in ALLOWED_POS
    ]

def load_all_texts_from_preprocessor(pre):
    texts = []
    try:
        pdfs = list(pre.load_all_pdfs())
    except Exception:
        pdfs = []
    try:
        txts = list(pre.load_txt())
    except Exception:
        txts = []
    for d in pdfs + txts:
        if hasattr(d, "page_content"):
            texts.append(d.page_content)
        elif isinstance(d, str):
            texts.append(d)
        else:
            texts.append(str(d))
    return texts

# --- TF only ---
def term_scores_tf(tokens: List[str]) -> List[Tuple[str, float, int]]:
    tf = Counter(tokens)
    return sorted([(t, float(tf[t]), tf[t]) for t in tf], key=lambda x: x[1], reverse=True)

# --- TF-IDF ---
def term_scores_tfidf(corpus_tokens: List[List[str]], song_tokens: List[str], song_index: int) -> List[Tuple[str, float, int]]:
    tokenized_texts = [" ".join(doc) for doc in corpus_tokens]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(tokenized_texts)
    feature_names = vectorizer.get_feature_names_out()

    tf = Counter(song_tokens)
    tfidf_scores = X[song_index].toarray().flatten()
    results = [(feature_names[i], tfidf_scores[i], tf.get(feature_names[i], 0)) 
               for i in range(len(feature_names)) if tfidf_scores[i] > 0]
    return sorted(results, key=lambda x: x[1], reverse=True)

# --- BM25 ---
def term_scores_bm25(corpus_tokens: List[List[str]], song_tokens: List[str], song_index: int) -> List[Tuple[str, float, int]]:
    bm25 = BM25Okapi(corpus_tokens)
    tf = Counter(song_tokens)
    unique_terms = list(tf.keys())
    scores = {}
    for term in unique_terms:
        scores[term] = float(bm25.get_scores([term])[song_index])
    return sorted([(t, scores[t], tf[t]) for t in scores], key=lambda x: x[1], reverse=True)

def compute_all_scores(pre, song_text: str):
    corpus_texts = load_all_texts_from_preprocessor(pre)
    corpus_with_song = corpus_texts + [song_text]
    song_index = len(corpus_with_song) - 1

    corpus_tokens = [lemmas_from_text(t) for t in corpus_with_song]
    song_tokens = corpus_tokens[song_index]

    global_tf = term_scores_tf(song_tokens)
    global_tfidf = term_scores_tfidf(corpus_tokens, song_tokens, song_index)
    global_bm25 = term_scores_bm25(corpus_tokens, song_tokens, song_index)

    local_corpus_tokens = [song_tokens]  
    local_tf = term_scores_tf(song_tokens)
    local_tfidf = term_scores_tfidf(local_corpus_tokens, song_tokens, 0)
    local_bm25 = term_scores_bm25(local_corpus_tokens, song_tokens, 0)

    return {
        "global_tf": global_tf,
        "global_tfidf": global_tfidf,
        "global_bm25": global_bm25,
        "local_tf": local_tf,
        "local_tfidf": local_tfidf,
        "local_bm25": local_bm25
    }


if __name__ == "__main__":
    pre = Preprocessor()
    sample_song = """сам сум качен на планината
во таа вечна потрага по суровината
суровоста на животот
и лакомоста твоја
е во тоа е вината
а не во виното
не се наоѓа таму вистината
се надевам дека 
не се плашиш од висината
не сакам да се чувствуваш
како да опишам и да сум културен
присилена

скокни, не е толку голема
висината
"""

    all_scores = compute_all_scores(pre, sample_song)

    first_n=40
    for key, scores in all_scores.items():
        print(f"\n--- {key.upper()} ---")
        for term, score, freq in scores[:first_n]:
            print(f"{term:<15} {score:>8.4f} {freq:>3}")
