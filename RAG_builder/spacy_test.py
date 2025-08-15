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
    # token.text → the exact word (surface form) from the text
    # token.lemma_ → the base form (dictionary form) of the word
    # token.pos_ → universal coarse Part-of-Speech tag (NOUN, VERB, ADJ, etc.)
    # token.tag_ → fine-grained POS tag (language-specific, e.g., gender, number, case)
    # token.dep_ → syntactic dependency label (relation to another word in the sentence)
    # token.shape_ → abstract pattern of the word’s characters (e.g., 'xxxx', 'Xxxx')
    # token.is_alpha → True if the token is made of only letters
    # token.is_stop → True if the token is a stopword (very common word with low meaning load)

    print(
        token.text,      # The actual text as it appears
        token.lemma_,    # Lemmatized (normalized) form
        token.pos_,      # Coarse POS tag
        token.tag_,      # Fine-grained POS tag
        token.dep_,      # Dependency relation label
        token.shape_,    # Shape of the token
        token.is_alpha,  # Is it only alphabetic characters?
        token.is_stop    # Is it considered a stopword?
    )
    

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
