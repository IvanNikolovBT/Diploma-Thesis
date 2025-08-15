from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import stop_words
import nltk
from nltk.tokenize import word_tokenize

# Sample text
text = "BM25 is a ranking function used in information retrieval. It improves upon TF-IDF by considering document length normalization."

# Tokenize and remove stopwords
tokens = [word for word in word_tokenize(text.lower()) if word.isalpha() and word not in stop_words.ENGLISH_STOP_WORDS]

# Create a "corpus" of a single document (the text itself)
corpus = [tokens]

# Train BM25
bm25 = BM25Okapi(corpus)

# Get BM25 scores for each word in the document
word_scores = {word: bm25.get_scores([word])[0] for word in tokens}

# Sort words by score (descending)
sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

# Top 5 keywords
top_keywords = [word for word, score in sorted_words[:5]]
print(top_keywords)