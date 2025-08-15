import math
from collections import Counter, defaultdict
from pprint import pprint
from preprocessor import Preprocessor  # Make sure this module is available
import spacy
from rank_bm25 import BM25Okapi

nlp = spacy.load("mk_core_news_lg")

# Load all documents for BM25 training
pre=Preprocessor()
all_docs = pre.load_all_pdfs() + pre.load_txt()
extra_texts = [doc.page_content for doc in all_docs]  # Extract text from Langchain Documents

def build_lemma_corpus(text, extra_texts):
    """Build lemma corpus with external documents for meaningful IDF"""
    docs = [text] + extra_texts
    lemma_docs = []
    for d in docs:
        doc = nlp(d)
        lemmas = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha
            and not token.is_stop
            and token.pos_ in {"NOUN", "PROPN", "ADJ", "VERB", "ADV"}
        ]
        lemma_docs.append(lemmas)  
    return lemma_docs

def compute_term_scores(lemma_docs):
    """Proper BM25 scoring with IDF calculation"""
    bm25 = BM25Okapi(lemma_docs)
    target_doc = lemma_docs[0]
    term_scores = {}
    
    for term in set(target_doc):
        if term in bm25.term_index:
            term_id = bm25.term_index[term]
            idf_term = bm25.idf[term_id]
        else:
            idf_term = 0  
        
        tf = target_doc.count(term)
        term_scores[term] = idf_term * (tf * (bm25.k1 + 1)) / (tf + bm25.k1 * (1 - bm25.b + bm25.b * len(target_doc)/bm25.avgdl))
    
    return term_scores

def extract_candidates(doc):
    lemma_to_tokens = defaultdict(list)
    for token in doc:
        if not token.is_alpha or token.is_stop:
            continue
        lemma_to_tokens[token.lemma_.lower()].append(token)
    return lemma_to_tokens

def representative_stats_for_lemma(tokens):
    """Handle missing POS tags and improve verb recognition"""
    if not tokens:
        return {
            "pos": "UNK", "dep": "", "children": 0, 
            "is_entity": False, "freq": 0, "tokens": []
        }
    
    pos_counts = Counter([t.pos_ for t in tokens])
    dep_counts = Counter([t.dep_ for t in tokens])
    rep_pos = pos_counts.most_common(1)[0][0] if pos_counts else "UNK"
    
    # Handle imperative verbs specifically
    if rep_pos == "UNK":
        if any(t.text.lower() in {"скокни", "стој", "бегај"} for t in tokens):
            rep_pos = "VERB"
    
    rep_dep = "ROOT" if any(t.dep_ == "ROOT" for t in tokens) else dep_counts.most_common(1)[0][0] if dep_counts else ""
    
    max_children = max(len(list(t.children)) for t in tokens) if tokens else 0
    is_entity = any(t.ent_type_ for t in tokens)
    
    return {
        "pos": rep_pos,
        "dep": rep_dep,
        "children": max_children,
        "is_entity": is_entity,
        "freq": len(tokens),
        "tokens": tokens,
    }

def normalize_dict(d):
    """Normalize with fallback for identical values"""
    if not d:
        return {}
    vals = [v for v in d.values() if v is not None]
    if not vals:
        return {k: 0.0 for k in d}
    mn, mx = min(vals), max(vals)
    if math.isclose(mn, mx):
        return {k: 1.0 for k in d}
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}

def score_lemmas(text, extra_texts, top_n=30):
    """Enhanced scoring with external corpus and verb prioritization"""
    lemma_docs = build_lemma_corpus(text, extra_texts)
    bm25_scores = compute_term_scores(lemma_docs)  # Use proper term scoring

    target_doc = nlp(text)
    lemma_tokens = extract_candidates(target_doc)

    lemma_info = {}
    for lemma, tokens in lemma_tokens.items():
        stats = representative_stats_for_lemma(tokens)
        lemma_info[lemma] = stats

    bm25_norm = normalize_dict({k: v for k, v in bm25_scores.items() if k in lemma_info})

    # Enhanced weights with verb priority
    pos_weight = {"PROPN": 2.0, "NOUN": 1.6, "VERB": 1.8, "ADJ": 1.3, "ADV": 0.9, "UNK": 0.5}
    dep_weight = {"ROOT": 2.0, "nsubj": 1.6, "obj": 1.4, "obl": 1.2, "amod": 1.1}

    pos_scores_raw = {}
    dep_scores_raw = {}
    child_counts_raw = {}
    sim_raw = {}

    for lemma, info in lemma_info.items():
        pos_raw = pos_weight.get(info["pos"], 0.8)
        dep_raw = dep_weight.get(info["dep"], 1.0)
        child_raw = info["children"]

        sim_val = 0.0
        for t in info["tokens"]:
            if t.has_vector and target_doc.has_vector:
                try:
                    s = t.similarity(target_doc)
                    if s > sim_val:
                        sim_val = s
                except:
                    continue

        pos_scores_raw[lemma] = pos_raw
        dep_scores_raw[lemma] = dep_raw
        child_counts_raw[lemma] = child_raw
        sim_raw[lemma] = max(0.0, sim_val) 

    pos_norm = normalize_dict(pos_scores_raw)
    dep_norm = normalize_dict(dep_scores_raw)
    child_norm = normalize_dict(child_counts_raw)
    sim_norm = normalize_dict(sim_raw)

    # Rebalanced weights (verbs get higher priority)
    weights = {
        "tfidf": 0.35,  
        "pos": 0.25,    # Increased importance
        "dep": 0.15,
        "entity": 0.05, # Reduced entity weight
        "children": 0.05,
        "sim": 0.15,    # Increased similarity weight
    }

    results = []
    for lemma, info in lemma_info.items():
        components = {
            "tfidf": bm25_norm.get(lemma, 0.0),
            "pos": pos_norm.get(lemma, 0.0),
            "dep": dep_norm.get(lemma, 0.0),
            "entity": 1.0 if info["is_entity"] else 0.0,
            "children": child_norm.get(lemma, 0.0),
            "sim": sim_norm.get(lemma, 0.0),
        }
        
        score = sum(weights[k] * comp for k, comp in components.items())

        results.append((
            lemma,
            {
                "score": score,
                "tfidf": components["tfidf"],
                "pos": info["pos"],
                "dep": info["dep"],
                "freq": info["freq"],
                "is_entity": info["is_entity"],
                "children": info["children"],
                **components
            }
        ))

    results.sort(key=lambda x: x[1]["score"], reverse=True)
    return results[:top_n]

if __name__ == "__main__":
    sample_text = """сам сум качен на планината
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
    ranked = score_lemmas(sample_text, extra_texts=extra_texts, top_n=40)

    print(f"{'Lemma':<20} {'Score':<7} {'TFIDF':<6} {'POS':<7} {'DEP':<8} {'Freq':<4} {'Ent':<5} {'Ch':<3}")
    print("-" * 70)
    for lemma, info in ranked:
        print(
            f"{lemma:<20} {info['score']:.3f}  {info['tfidf']:.3f}  {info['pos']:<7} {info['dep']:<8} {info['freq']:<4} {str(info['is_entity']):<5} {info['children']:<3}"
        )

    print("\nTop lemmas (ordered):")
    print([lemma for lemma, _ in ranked])