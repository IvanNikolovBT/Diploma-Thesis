import math
from collections import Counter, defaultdict
from pprint import pprint

import spacy
from rank_bm25 import BM25Okapi

nlp = spacy.load("mk_core_news_lg")

def build_lemma_corpus(text, extra_texts=None):
    docs = [text] + (extra_texts or [])
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

def bm25_on_lemmas(lemma_docs):
    bm25 = BM25Okapi(lemma_docs)
    target_doc = lemma_docs[0]
    scores = bm25.get_scores(target_doc)
    
    # REVERTED: Original approach of assigning same score to all tokens
    lemma_scores = dict(zip(target_doc, [scores[0]] * len(target_doc)))
    return lemma_scores

def extract_candidates(doc):
    lemma_to_tokens = defaultdict(list)
    for token in doc:
        if not token.is_alpha:
            continue
        lemma_to_tokens[token.lemma_.lower()].append(token)
    return lemma_to_tokens

def representative_stats_for_lemma(tokens):
    pos_counts = Counter([t.pos_ for t in tokens])
    dep_counts = Counter([t.dep_ for t in tokens])
    rep_pos = pos_counts.most_common(1)[0][0]
    if any(t.dep_ == "ROOT" or t.dep_ == "root" for t in tokens):
        rep_dep = "ROOT"
    else:
        rep_dep = dep_counts.most_common(1)[0][0]
    max_children = max(len(list(t.children)) for t in tokens)
    is_entity = any(t.ent_type_ for t in tokens)
    freq = len(tokens)
    return {
        "pos": rep_pos,
        "dep": rep_dep,
        "children": max_children,
        "is_entity": is_entity,
        "freq": freq,
        "tokens": tokens,
    }

def normalize_dict(d):
    if not d:
        return {}
    vals = list(d.values())
    mn, mx = min(vals), max(vals)
    if math.isclose(mn, mx):
        return {k: 1.0 for k in d}
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}

def score_lemmas(text, extra_texts=None, top_n=30):
    lemma_docs = build_lemma_corpus(text, extra_texts)
    bm25_scores = bm25_on_lemmas(lemma_docs)

    target_doc = nlp(text)
    lemma_tokens = extract_candidates(target_doc)

    lemma_info = {}
    for lemma, tokens in lemma_tokens.items():
        stats = representative_stats_for_lemma(tokens)
        lemma_info[lemma] = stats

    bm25_for_target = {k: v for k, v in bm25_scores.items() if k in lemma_info}
    bm25_norm = normalize_dict(bm25_for_target)

    pos_weight = {"PROPN": 2.0, "NOUN": 1.6, "ADJ": 1.3, "VERB": 1.0, "ADV": 0.9}
    dep_weight = {"ROOT": 2.0, "nsubj": 1.6, "obj": 1.4, "obl": 1.2, "amod": 1.1}

    pos_scores_raw = {}
    dep_scores_raw = {}
    child_counts_raw = {}
    sim_raw = {}

    for lemma, info in lemma_info.items():
        pos_raw = pos_weight.get(info["pos"], 1.0)
        dep_raw = dep_weight.get(info["dep"], 1.0)
        child_raw = info["children"]

        sim_val = 0.0
        for t in info["tokens"]:
            try:
                if t.has_vector and target_doc.has_vector:
                    s = t.similarity(target_doc)
                    if s > sim_val:
                        sim_val = s
            except Exception:
                continue

        sim_val = max(0.0, sim_val)
        pos_scores_raw[lemma] = pos_raw
        dep_scores_raw[lemma] = dep_raw
        child_counts_raw[lemma] = child_raw
        sim_raw[lemma] = sim_val

    pos_norm = normalize_dict(pos_scores_raw)
    dep_norm = normalize_dict(dep_scores_raw)
    child_norm = normalize_dict(child_counts_raw)
    sim_norm = normalize_dict(sim_raw)

    weights = {
        "bm25": 0.50,
        "pos": 0.15,
        "dep": 0.15,
        "entity": 0.08,
        "children": 0.05,
        "sim": 0.07,
    }

    results = []
    for lemma, info in lemma_info.items():
        bm25_component = bm25_norm.get(lemma, 0.0)
        pos_component = pos_norm.get(lemma, 0.0)
        dep_component = dep_norm.get(lemma, 0.0)
        entity_component = 1.0 if info["is_entity"] else 0.0
        children_component = child_norm.get(lemma, 0.0)
        sim_component = sim_norm.get(lemma, 0.0)

        score = (
            weights["bm25"] * bm25_component +
            weights["pos"] * pos_component +
            weights["dep"] * dep_component +
            weights["entity"] * entity_component +
            weights["children"] * children_component +
            weights["sim"] * sim_component
        )

        results.append(
            (
                lemma,
                {
                    "score": score,
                    "bm25": bm25_component,
                    "pos": info["pos"],
                    "dep": info["dep"],
                    "freq": info["freq"],
                    "is_entity": info["is_entity"],
                    "children": info["children"],
                    "pos_component": pos_component,
                    "dep_component": dep_component,
                    "children_component": children_component,
                    "sim_component": sim_component,
                },
            )
        )

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
    ranked = score_lemmas(sample_text, extra_texts=None, top_n=40)

    print(f"{'Lemma':<20} {'Score':<7} {'BM25':<6} {'POS':<7} {'DEP':<8} {'Freq':<4} {'Ent':<3} {'Ch':<3}")
    print("-" * 70)
    for lemma, info in ranked:
        print(
            f"{lemma:<20} {info['score']:.3f}  {info['bm25']:.3f}  {info['pos']:<7} {info['dep']:<8} {info['freq']:<4} {str(info['is_entity']):<3} {info['children']:<3}"
        )

    top_lemmas = [l for l, _ in ranked]
    print("\nTop lemmas (ordered):")
    pprint(top_lemmas)