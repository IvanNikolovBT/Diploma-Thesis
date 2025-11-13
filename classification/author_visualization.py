import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from adjustText import adjust_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.sparse import hstack

class SongStyleVisualizer:
    DEFAULT_CSV_PATH = "/home/ivan/Desktop/Diplomska/final_results_csv/cleaned_songs_with_perplexity.csv"
    DEFAULT_STYLE_CSV_PATH = "/home/ivan/Desktop/Diplomska/api_styles_all_in_one_text.csv"

    MODEL_NAME = "macedonizer/mk-roberta-base"
    tokenizer = None
    model = None
    device = None

    @classmethod
    def _init_transformer(cls):
        if cls.tokenizer is None:
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.MODEL_NAME)
            if cls.tokenizer.pad_token is None:
                cls.tokenizer.pad_token = cls.tokenizer.eos_token
            cls.model = AutoModel.from_pretrained(cls.MODEL_NAME)
            cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {cls.device}")
            cls.model.to(cls.device)
            cls.model.eval()

    def __init__(self, csv_path=None, style_csv_path=None):
        self.csv_path = csv_path or self.DEFAULT_CSV_PATH
        self.style_csv_path = style_csv_path or self.DEFAULT_STYLE_CSV_PATH

        print(f"Loading lyrics from: {self.csv_path}")
        self.lyrics_df = pd.read_csv(self.csv_path)
        self.lyrics_df["song_text"] = self.lyrics_df["song_text"].fillna("").astype(str)

        print(f"Loading styles from: {self.style_csv_path}")
        self.styles_df = pd.read_csv(self.style_csv_path)
        self.styles_df["extracted_styles"] = self.styles_df["extracted_styles"].fillna("").astype(str)

        self.authors = None
        self._embedding_cache = {} 

        
        self._init_transformer()

    def _sample_per_author(self, df, max_author_songs):
        if max_author_songs is None:
            return df.copy()
        sampled = []
        for author in df["author"].unique():
            songs = df[df["author"] == author]
            if len(songs) > max_author_songs:
                songs = songs.sample(n=max_author_songs, random_state=42)
            sampled.append(songs)
        return pd.concat(sampled).reset_index(drop=True)

    def _reduce_dimensions(self, data, use_pca=False):
        reducer = PCA(n_components=2) if use_pca else TSNE(n_components=2, perplexity=20, learning_rate=100, init="random", random_state=42)
        X = data.toarray() if hasattr(data, "toarray") else data
        return reducer.fit_transform(X)

    def _add_labels_with_clusters(self, df, xcol, ycol, label_clusters, min_points=5, eps=0.5):
        if not label_clusters or self.authors is None:
            return []
        texts = []
        for author in self.authors:
            points = df[df["author"] == author][[xcol, ycol]].values
            if len(points) < min_points:
                continue
            clustering = DBSCAN(eps=eps, min_samples=3).fit(points)
            labels = clustering.labels_
            noise_mask = labels != -1
            if not noise_mask.any():
                cx, cy = points.mean(axis=0)
            else:
                uniq, counts = np.unique(labels[noise_mask], return_counts=True)
                largest = uniq[np.argmax(counts)]
                cx, cy = points[labels == largest].mean(axis=0)
            jitter = np.random.uniform(-0.05, 0.05, 2)
            txt = plt.text(cx + jitter[0], cy + jitter[1], author, fontsize=10, weight="bold",
                           ha="center", va="center", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
            texts.append(txt)
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        return texts

    def _extract_style_features(self, df):
        features = []
        pattern = r"(.+?):\s*(.+)"
        for styles in df["extracted_styles"]:
            vec = [0] * 86
            for i, (_, val) in enumerate(re.findall(pattern, styles)):
                if i >= 86: break
                if val.strip().lower().startswith("да"):
                    vec[i] = 1
            features.append(vec)
        return np.array(features)

    def _get_content_matrix(self, df, use_transformer=False):
        df = self._sample_per_author(df, None)  
        texts = df["song_text"].tolist()
        authors = df["author"].tolist()
        cache_key = (hash(tuple(texts)), use_transformer)

        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        if use_transformer:
            print("Computing transformer embeddings...")
            emb = self._get_transformer_embeddings(texts)
        else:
            vectorizer = TfidfVectorizer(max_features=5000)
            emb = vectorizer.fit_transform(texts).toarray()

        self._embedding_cache[cache_key] = (emb, authors)
        return emb, authors

    def _get_transformer_embeddings(self, texts, batch_size=8, max_len=512):
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                max_length=max_len,
                padding='max_length',
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state
                mask = attention_mask.unsqueeze(-1).float()
                summed = (hidden * mask).sum(1)
                lengths = mask.sum(1).clamp(min=1.0)
                mean_pooled = summed / lengths
                all_emb.append(mean_pooled.cpu().numpy())
        return np.vstack(all_emb)

    def get_distances(self, X):
        sim = cosine_similarity(X)
        n = sim.shape[0]
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(1 - sim[i, j])
        return np.array(dists)

    # ===================================================================
    # VISUALIZATION & ANALYSIS METHODS
    # ===================================================================
    def plot_all_songs(self, use_pca=False, max_author_songs=None, label_clusters=False, use_transformer=False):
        df = self._sample_per_author(self.lyrics_df, max_author_songs)
        self.authors = df["author"].unique()

        if use_transformer:
            X = self._get_transformer_embeddings(df["song_text"].tolist())
        else:
            vectorizer = TfidfVectorizer(max_features=5000)
            X = vectorizer.fit_transform(df["song_text"]).toarray()

        reduced = self._reduce_dimensions(X, use_pca)
        xcol, ycol = ("pca_x", "pca_y") if use_pca else ("tsne_x", "tsne_y")
        df[xcol], df[ycol] = reduced[:, 0], reduced[:, 1]

        title = f"{'Transformer' if use_transformer else 'TF-IDF'} Songs → 2D ({'PCA' if use_pca else 't-SNE'})"
        plt.figure(figsize=(12, 8))
        for author in self.authors:
            mask = df["author"] == author
            plt.scatter(df.loc[mask, xcol], df.loc[mask, ycol], label=author, s=25, alpha=0.7)
        self._add_labels_with_clusters(df, xcol, ycol, label_clusters, eps=1.0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(title); plt.xlabel(xcol); plt.ylabel(ycol)
        plt.tight_layout(); plt.show()

    def plot_authors_aggregate(self, use_pca=False, use_transformer=False):
        df = self.lyrics_df.copy()
        if use_transformer:
            author_texts = df.groupby("author")["song_text"].apply(list).reset_index()
            all_emb = []
            all_authors = []
            for _, row in author_texts.iterrows():
                emb = self._get_transformer_embeddings(row["song_text"])
                all_emb.append(emb.mean(axis=0))
                all_authors.append(row["author"])
            X = np.array(all_emb)
        else:
            author_texts = df.groupby("author")["song_text"].apply(" ".join).reset_index()
            X = TfidfVectorizer(max_features=5000).fit_transform(author_texts["song_text"]).toarray()

        reduced = self._reduce_dimensions(X, use_pca)
        author_texts[["x", "y"]] = reduced if not use_transformer else pd.DataFrame(reduced, index=all_authors)[[0, 1]]

        title = f"Authors {'Transformer' if use_transformer else 'TF-IDF'} → 2D ({'PCA' if use_pca else 't-SNE'})"
        plt.figure(figsize=(12, 8))
        plt.scatter(author_texts["x"], author_texts["y"], s=60, alpha=0.7)
        for _, row in author_texts.iterrows():
            plt.text(row["x"], row["y"], row["author"], fontsize=9, weight="bold",
                     ha="center", va="center", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5))
        plt.title(title); plt.xlabel("Component 1"); plt.ylabel("Component 2")
        plt.tight_layout(); plt.show()

    def plot_author_styles(self, use_pca=False, max_author_songs=None, label_clusters=False):
        df = self._sample_per_author(self.styles_df, max_author_songs)
        self.authors = df["author"].unique()
        features = self._extract_style_features(df)
        reduced = self._reduce_dimensions(features, use_pca)
        xcol, ycol = ("pca_x", "pca_y") if use_pca else ("tsne_x", "tsne_y")
        df[xcol], df[ycol] = reduced[:, 0], reduced[:, 1]
        title = f"Author Styles → 2D ({'PCA' if use_pca else 't-SNE'})"
        plt.figure(figsize=(12, 8))
        for author in self.authors:
            mask = df["author"] == author
            plt.scatter(df.loc[mask, xcol], df.loc[mask, ycol], label=author, s=25, alpha=0.7)
        self._add_labels_with_clusters(df, xcol, ycol, label_clusters, eps=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(title); plt.xlabel(xcol); plt.ylabel(ycol)
        plt.tight_layout(); plt.show()

    def plot_author_aggregate_styles(self, use_pca=False, max_author_songs=None, label_points=True):
        df = self._sample_per_author(self.styles_df, max_author_songs)
        features = self._extract_style_features(df)
        df_feat = pd.DataFrame(features)
        df_feat["author"] = df["author"].values
        aggregated = df_feat.groupby("author").max()
        reduced = self._reduce_dimensions(aggregated.values, use_pca)
        xcol, ycol = ("pca_x", "pca_y") if use_pca else ("tsne_x", "tsne_y")
        aggregated[xcol], aggregated[ycol] = reduced[:, 0], reduced[:, 1]
        title = f"Aggregated Author Styles → 2D ({'PCA' if use_pca else 't-SNE'})"
        plt.figure(figsize=(12, 8))
        plt.scatter(aggregated[xcol], aggregated[ycol], s=50, alpha=0.8)
        if label_points:
            texts = []
            for author, row in aggregated.iterrows():
                txt = plt.text(row[xcol], row[ycol], author, fontsize=10, weight="bold",
                               ha="center", va="center", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
                texts.append(txt)
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        plt.title(title); plt.xlabel(xcol); plt.ylabel(ycol)
        plt.tight_layout(); plt.show()
    def _get_neighbors(self, sim_matrix, idx, k, author_list):
        scores = sim_matrix[idx].copy()
        scores[idx] = -1
        neighbors_idx = np.argsort(scores)[-k:][::-1]
        return [author_list[i] for i in neighbors_idx]
    def distance_rank_correlation(self, max_author_songs=None, use_transformer=False):
        df_lyric = self._sample_per_author(self.lyrics_df, max_author_songs)
        if use_transformer:
            Xc = self._get_transformer_embeddings(df_lyric["song_text"].tolist())
            df_c = pd.DataFrame(Xc, index=df_lyric["author"])
            Xc_agg = df_c.groupby(level=0).mean().values
        else:
            txt = df_lyric.groupby('author')['song_text'].apply(' '.join).reset_index()
            Xc_agg = TfidfVectorizer(max_features=5000).fit_transform(txt['song_text']).toarray()
        dc = self.get_distances(Xc_agg)

        df_style = self._sample_per_author(self.styles_df, max_author_songs)
        Xs = self._extract_style_features(df_style)
        df_s = pd.DataFrame(Xs); df_s['author'] = df_style['author'].values
        Xs_agg = df_s.groupby('author').max().values
        ds = self.get_distances(Xs_agg)

        rho, p = spearmanr(dc, ds)
        print(f"Spearman ρ = {rho:.3f} (p = {p:.2e}) – {'Transformer' if use_transformer else 'TF-IDF'} vs Style")
        if rho > 0.5: print("Strong")
        elif rho > 0.2: print("Moderate")
        else: print("Weak")
        return rho, p


    def plot_similarity_heatmap(
        self,
        use_transformer=False,
        max_author_songs=None,
        cmap='viridis',
        remove_stopwords=False,
        custom_stop_words=None
    ):
        default_stop_words = [
            'ќе', 'би', 'ко', 'го', 'како', 'ги', 'ми', 'ти', 'те', 'му', 'само',
            'зашто', 'таа', 'тие', 'нѐ', 'но', 'сѐ', 'со', 'по', 'ли', 'ој', 'ни',
            'ниту', 'pinterest', 'до', 'таа', 'ние', 'вие', 'тие', 'си', 'то', 'сме',
            'бил', 'јас', 'нека', 'кога', 'колку', 'тоа', 'дека', 'или', 'зар', 'ил',
            'ме', 'со', 'кој', 'кон', 'та', 'оваа', 'овој', 'тој', 'кај', 'се', 'туку',
            'ние', 'вие', 'тие', 'нѐ'
        ]
        stop_words = custom_stop_words if custom_stop_words is not None else default_stop_words
        df = self._sample_per_author(self.lyrics_df, max_author_songs)
        if use_transformer:
            X = self._get_transformer_embeddings(df["song_text"].tolist())
            df_x = pd.DataFrame(X, index=df["author"])
            X = df_x.groupby(level=0).mean().values
            authors = df_x.index.unique().tolist()
        else:
            texts = df.groupby('author')['song_text'].apply(' '.join).reset_index()
            kwargs = {"max_features": 5000, "token_pattern": r"(?u)\b\w\w+\b"}
            if remove_stopwords:
                kwargs["stop_words"] = stop_words
            tfidf = TfidfVectorizer(**kwargs)
            X = tfidf.fit_transform(texts['song_text']).toarray()
            authors = texts['author'].tolist()
        sim = cosine_similarity(X)
        n = len(authors)
        print("\n=== Per-Author Similarity Rankings ===")
        biggest_diff = 1.0
        biggest_diff_author = ""
        for i, author in enumerate(authors):
            sims = [(authors[j], sim[i, j]) for j in range(n) if j != i]
            sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
            max_author, max_sim = sims_sorted[0]
            min_author, min_sim = sims_sorted[-1]
            if min_sim < biggest_diff:
                biggest_diff = min_sim
                biggest_diff_author = f'Author {author} has the biggest difference with {min_author} - {1 - min_sim:.4f}'
            biggest_drop = max_sim - min_sim
            print(f"\n--- {author} ---")
            print(f" Biggest similarity drop: {biggest_drop:.4f}")
            print(f" From most similar: {max_author} ({max_sim:.4f})")
            print(f" To least similar: {min_author} ({min_sim:.4f})")
        print(biggest_diff_author)
        print("\n=============================================\n")
        plt.figure(figsize=(10, 8))
        im = plt.imshow(sim, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(im, label='Cosine similarity')
        plt.xticks(range(n), authors, rotation=90, fontsize=8)
        plt.yticks(range(n), authors, fontsize=8)
        title = 'Author similarity (transformer)' if use_transformer else 'Author similarity (TF-IDF)'
        plt.title(title)
        plt.tight_layout()
        plt.show()
    def _aggregate_author_embeddings(
        self,
        df: pd.DataFrame,
        col:str,
        max_author_songs: int | None,
        use_transformer: bool,
        tfidf_max_features: int = 5000
    ) -> np.ndarray:

        df_s = self._sample_per_author(df, max_author_songs)
        authors = df_s["author"].unique()

        if use_transformer:
            embs = self._get_transformer_embeddings(df_s[col].tolist())
            df_e = pd.DataFrame(embs, index=df_s["author"])
            X = df_e.groupby(level=0).mean().reindex(authors).values
        else:
            txt = (
                df_s.groupby("author")[col]
                .apply(" ".join)
                .reindex(authors)
                .fillna("")
            )
            vec = TfidfVectorizer(max_features=tfidf_max_features)
            X = vec.fit_transform(txt).toarray()

        return X
    
    def compare_two_content_csvs(
            self,
            csv1_path: str,
            csv2_path: str,
            use_transformer: bool = False,
            max_author_songs: int | None = None,
            tfidf_max_features: int = 5000
        ) -> tuple[float, float]:

            df1 = pd.read_csv(csv1_path)
            df2 = pd.read_csv(csv2_path)

            X1 = self._aggregate_author_embeddings(
                df1,'song_text',max_author_songs, use_transformer, tfidf_max_features
            )
            X2 = self._aggregate_author_embeddings(
                df2,'new_song', max_author_songs, use_transformer, tfidf_max_features
            )


            authors1 = df1["author"].unique().tolist()
            authors2 = df2["author"].unique().tolist()
            common = sorted(set(authors1) & set(authors2))
            if len(common) < 2:
                raise ValueError(f"Only {len(common)} common authors – need ≥2.")

            idx1 = [authors1.index(a) for a in common]
            idx2 = [authors2.index(a) for a in common]

            X1_aligned = X1[idx1]
            X2_aligned = X2[idx2]


            dist1 = pdist(X1_aligned, metric="cosine")
            dist2 = pdist(X2_aligned, metric="cosine")

            rho, p = spearmanr(dist1, dist2)

            method = "Transformer" if use_transformer else "TF-IDF"
            print(f"\nCOMPARE: {csv1_path.split('/')[-1]}  vs  {csv2_path.split('/')[-1]}")
            print(f"   Method: {method}")
            print(f"   Common authors: {len(common)}")
            print(f"   Pairwise distances: {len(dist1)}")
            print(f"   Spearman rho = {rho:.3f}  (p = {p:.2e})")
            if rho > 0.5:
                print("   -> Strong agreement")
            elif rho > 0.2:
                print("   -> Moderate agreement")
            else:
                print("   -> Weak agreement")

            return rho, p

import os
test=SongStyleVisualizer()
def get_bertscore_files_sorted(directory: str):

    if not os.path.isdir(directory):
        raise ValueError(f"Directory not found: {directory}")

    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith("bertscore.csv") and os.path.isfile(os.path.join(directory, f))
    ]
    
    return sorted(files)   
files= get_bertscore_files_sorted("/home/ivan/Desktop/Diplomska/final_results_csv")
print('TF IDF COMPARISON')
for file in files:
    cleaned_csv='/home/ivan/Desktop/Diplomska/final_results_csv/cleaned_songs_with_perplexity.csv'
    test.compare_two_content_csvs(cleaned_csv,file)
print('MACEDONIZER COMPARISON')
for file in files:
    cleaned_csv='/home/ivan/Desktop/Diplomska/final_results_csv/cleaned_songs_with_perplexity.csv'
    test.compare_two_content_csvs(cleaned_csv,file,use_transformer=True)