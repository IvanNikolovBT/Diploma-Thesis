import spacy
from collections import Counter

nlp = spacy.load("mk_core_news_lg")

text = """сам сум качен на планината
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

doc = nlp(text)

noun_chunks = []
current_chunk = []

for token in doc:
    print(f'token {token}')
    if token.pos_ in {"NOUN", "PROPN", "ADJ"}:
        current_chunk.append(token.text)
    else:
        if current_chunk:
            noun_chunks.append(" ".join(current_chunk))
            current_chunk = []
if current_chunk:
    noun_chunks.append(" ".join(current_chunk))

print("Noun phrases:", noun_chunks)


keywords = [
    token.lemma_ for token in doc
    if token.pos_ in {"NOUN", "PROPN", "ADJ", "VERB"}
    and not token.is_stop
    and token.is_alpha
]
freq = Counter(keywords)
print("Most common keywords:", freq.most_common(10))
