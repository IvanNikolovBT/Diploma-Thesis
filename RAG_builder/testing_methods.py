
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


def term_scores_tf(tokens: List[str]) -> List[Tuple[str, float, int]]:
    tf = Counter(tokens)
    return sorted([(t, float(tf[t]), tf[t]) for t in tf], key=lambda x: x[1], reverse=True)

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
def compute_best_ones(pre, song_text: str):
    corpus_texts = load_all_texts_from_preprocessor(pre)
    corpus_with_song = corpus_texts + [song_text]
    song_index = len(corpus_with_song) - 1

    corpus_tokens = [lemmas_from_text(t) for t in corpus_with_song]
    song_tokens = corpus_tokens[song_index]

   
    global_tfidf = term_scores_tfidf(corpus_tokens, song_tokens, song_index)
    global_bm25 = term_scores_bm25(corpus_tokens, song_tokens, song_index)

    return {

        "global_tfidf": global_tfidf,
        "global_bm25": global_bm25
    }

def normalize_column(values):
        mn, mx = min(values), max(values)
        if mx - mn < 1e-9:
            return [1.0] * len(values)  # all same
        return [(v - mn) / (mx - mn) for v in values]
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

    all_scores = compute_best_ones(pre, sample_song)

    first_n=40
    for key, scores in all_scores.items():
        print(f"\n--- {key.upper()} ---")
        for term, score, freq in scores[:first_n]:
            print(f"{term:<15} {score:>8.4f} {freq:>3}")
    with open('testing_methods.csv',mode='w') as f:
        global_tf_idf={}
        global_bm={}
        for key,scores in all_scores.items():
            if key=='global_tfidf':
                for term,score,freq in scores[:first_n]:
                    global_tf_idf[term]={"score":score,'freq':freq}
            else:
                for term,score,freq in scores[:first_n]:
                    global_bm[term]={"score":score,'freq':freq}            

        for key,value in global_bm.items():
            pass

#glboal tf, alot of the same result, bad.
#global tfidf, good results, different values for each word
#global bm25, good results, diferent values for each word.


#local tf bad, alot of the same result for the words
#local tbad, alot of the same result for the words
#local bm25 bad, alot of the same result for the words
#local tfidf bad, alot of the same result for the words

#from these we can chose between globlal tfidf or global bm25 for searching the words. 
#will make tests on them 

"""--- GLOBAL_TF ---
вина              2.0000   2
висина            2.0000   2
качен             1.0000   1
планина           1.0000   1
вечна             1.0000   1
потрага           1.0000   1
суровина          1.0000   1
суровост          1.0000   1
живот             1.0000   1
лакомоста         1.0000   1
наоѓа             1.0000   1
вистина           1.0000   1
надева            1.0000   1
плаши             1.0000   1
сака              1.0000   1
чувствува         1.0000   1
опише             1.0000   1
културен          1.0000   1
присилен          1.0000   1
толку             1.0000   1
голем             1.0000   1

--- GLOBAL_TFIDF ---
висина            0.4396   2
лакомоста         0.2811   1
присилен          0.2811   1
вина              0.2636   2
суровина          0.2584   1
суровост          0.2584   1
опише             0.2357   1
потрага           0.2224   1
качен             0.2130   1
културен          0.2057   1
надева            0.1997   1
наоѓа             0.1814   1
вистина           0.1806   1
вечна             0.1799   1
чувствува         0.1603   1
плаши             0.1583   1
планина           0.1395   1
толку             0.1275   1
голем             0.1006   1
сака              0.1005   1
живот             0.0919   1

--- GLOBAL_BM25 ---
лакомоста        11.7381   1
присилен         11.7381   1
висина           10.7375   2
суровина         10.4736   1
суровост         10.4736   1
опише             9.3354   1
потрага           8.6964   1
качен             8.2497   1
културен          7.9058   1
надева            7.6259   1
наоѓа             6.7709   1
вистина           6.7350   1
вечна             6.6999   1
чувствува         5.7867   1
плаши             5.6938   1
вина              5.5615   2
планина           4.8071   1
толку             4.2312   1
голем             2.8905   1
сака              2.8871   1
живот             2.4288   1

--- LOCAL_TF ---
вина              2.0000   2
висина            2.0000   2
качен             1.0000   1
планина           1.0000   1
вечна             1.0000   1
потрага           1.0000   1
суровина          1.0000   1
суровост          1.0000   1
живот             1.0000   1
лакомоста         1.0000   1
наоѓа             1.0000   1
вистина           1.0000   1
надева            1.0000   1
плаши             1.0000   1
сака              1.0000   1
чувствува         1.0000   1
опише             1.0000   1
културен          1.0000   1
присилен          1.0000   1
толку             1.0000   1
голем             1.0000   1

--- LOCAL_TFIDF ---
вина              0.3849   2
висина            0.3849   2
вечна             0.1925   1
вистина           0.1925   1
голем             0.1925   1
живот             0.1925   1
качен             0.1925   1
културен          0.1925   1
лакомоста         0.1925   1
надева            0.1925   1
наоѓа             0.1925   1
опише             0.1925   1
планина           0.1925   1
плаши             0.1925   1
потрага           0.1925   1
присилен          0.1925   1
сака              0.1925   1
суровина          0.1925   1
суровост          0.1925   1
толку             0.1925   1
чувствува         0.1925   1

--- LOCAL_BM25 ---
качен            -0.2747   1
планина          -0.2747   1
вечна            -0.2747   1
потрага          -0.2747   1
суровина         -0.2747   1
суровост         -0.2747   1
живот            -0.2747   1
лакомоста        -0.2747   1
наоѓа            -0.2747   1
вистина          -0.2747   1
надева           -0.2747   1
плаши            -0.2747   1
сака             -0.2747   1
чувствува        -0.2747   1
опише            -0.2747   1
културен         -0.2747   1
присилен         -0.2747   1
толку            -0.2747   1
голем            -0.2747   1
вина             -0.3924   2
висина           -0.3924   2"""