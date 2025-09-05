#!/usr/bin/env python3
import math
from collections import Counter
from typing import List, Tuple

import spacy
from rank_bm25 import BM25Okapi
from preprocessor import Preprocessor

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
    # adapt to Preprocessor return types: objects with .page_content or plain strings
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

def term_scores_using_bm25(pre, song_text: str, include_song_in_corpus: bool = True,
                           normalize: bool = False) -> List[Tuple[str, float, int]]:
    """
    Uses the imported BM25Okapi implementation directly.
    Returns list of (term, score_for_song_doc, frequency_in_song), sorted descending by score.
    """
    corpus_texts = load_all_texts_from_preprocessor(pre)
    # include song in corpus (so BM25 stats reflect it). If you want to exclude song from IDF, set False.
    if include_song_in_corpus:
        corpus_texts = corpus_texts + [song_text]

    # tokenized corpus (list of lemma lists)
    tokenized = [lemmas_from_text(t) for t in corpus_texts]

    # build BM25 model from the tokenized corpus
    bm25 = BM25Okapi(tokenized)

    # song is the last doc if included; otherwise we still compute lemmas_from_text(song_text)
    song_tokens = tokenized[-1] if include_song_in_corpus else lemmas_from_text(song_text)
    song_index = (len(tokenized) - 1) if include_song_in_corpus else None

    if not song_tokens:
        return []

    tf = Counter(song_tokens)
    unique_terms = list(tf.keys())

    term_scores = {}
    # For each term, query BM25 (single-term query) and take the score for the song document
    # This uses bm25.get_scores([...]) from the imported library directly.
    for term in unique_terms:
        try:
            scores = bm25.get_scores([term])
            # if song is in corpus we take the song_index element, otherwise take aggregated score across tokenized docs:
            if song_index is not None:
                score_for_song = float(scores[song_index])
            else:
                # if song not in corpus, approximate term's relevance to the song by summing scores for docs
                # where the term appears AND weighting by term freq in song (fallback)
                # but in typical usage include_song_in_corpus=True is preferred.
                score_for_song = float(sum(scores)) * (tf[term] / sum(tf.values()))
        except Exception:
            score_for_song = 0.0

        term_scores[term] = score_for_song

    # optional normalization to [0,1]
    if normalize and term_scores:
        vals = list(term_scores.values())
        mn, mx = min(vals), max(vals)
        if math.isclose(mn, mx):
            term_scores = {k: 1.0 for k in term_scores}
        else:
            term_scores = {k: (v - mn) / (mx - mn) for k, v in term_scores.items()}

    # build final sorted list: (term, score, freq)
    sorted_terms = sorted(
        [(t, term_scores[t], tf[t]) for t in term_scores],
        key=lambda x: x[1],
        reverse=True,
    )
    return sorted_terms

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

    results = term_scores_using_bm25(pre, sample_song, include_song_in_corpus=True, normalize=False)

    print(f"{'term':<20} {'score':>12} {'freq':>6}")
    print("-" * 40)
    for term, score, freq in results:
        print(f"{term:<20} {score:12.6f} {freq:6d}")
