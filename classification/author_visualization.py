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
from scipy.linalg import orthogonal_procrustes
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
        df = self._sample_per_author(df, None)  # Use all songs
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
    def compare_content_vs_style_similarity(self, top_k=5, max_author_songs=None, use_transformer=False):
       
        lyrics_sample = self._sample_per_author(self.lyrics_df, max_author_songs)
        if use_transformer:
            Xc = self._get_transformer_embeddings(lyrics_sample["song_text"].tolist())
            df_c = pd.DataFrame(Xc, index=lyrics_sample["author"])
            content_agg = df_c.groupby(level=0).mean().values
        else:
            author_texts = lyrics_sample.groupby("author")["song_text"].apply(" ".join).reset_index()
            content_agg = TfidfVectorizer(max_features=5000).fit_transform(author_texts["song_text"]).toarray()

        content_sim = cosine_similarity(content_agg)

        
        styles_sample = self._sample_per_author(self.styles_df, max_author_songs)
        style_features = self._extract_style_features(styles_sample)
        df_style = pd.DataFrame(style_features)
        df_style["author"] = styles_sample["author"].values
        style_agg = df_style.groupby("author").max().values
        style_sim = cosine_similarity(style_agg)

        
        authors_c = author_texts["author"].tolist() if not use_transformer else lyrics_sample.groupby("author").apply(lambda x: x.name).index.tolist()
        authors_s = df_style.groupby("author").max().index.tolist()
        common = sorted(set(authors_c) & set(authors_s))
        idx_c = [authors_c.index(a) for a in common]
        idx_s = [authors_s.index(a) for a in common]
        content_sim = content_sim[np.ix_(idx_c, idx_c)]
        style_sim = style_sim[np.ix_(idx_s, idx_s)]



        results = {}
        overlaps = []
        for author in common:
            idx = common.index(author)
            c_n = self._get_neighbors(content_sim, idx, top_k)
            s_n = self._get_neighbors(style_sim, idx, top_k)
            overlap = len(set(c_n) & set(s_n))
            jaccard = overlap / (2 * top_k)
            results[author] = {"content_neighbors": c_n, "style_neighbors": s_n, "overlap_count": overlap, "jaccard_alignment": jaccard}
            overlaps.append(jaccard)

        mean_j = np.mean(overlaps)
        print(f"\nCONTENT ({'Transformer' if use_transformer else 'TF-IDF'}) vs STYLE – Top-{top_k} NN")
        print(f"{'Author':<25} {'Overlap':<8} {'Jaccard':<8} NN → NN")
        print("-" * 80)
        for a in sorted(results, key=lambda x: results[x]["jaccard_alignment"], reverse=True):
            r = results[a]
            print(f"{a:<25} {r['overlap_count']:<8} {r['jaccard_alignment']:.3f} "
                  f"{', '.join(r['content_neighbors'][:3])}{'...' if len(r['content_neighbors'])>3 else ''} → "
                  f"{', '.join(r['style_neighbors'][:3])}{'...' if len(r['style_neighbors'])>3 else ''}")
        print(f"\nMean Jaccard = {mean_j:.3f}")
        return results, mean_j
    def compare_author_representations(self, max_author_songs=None, use_transformer=True, n_components=50):
        """
        Compare TF-IDF vs Transformer using *separate* PCA reductions.
        More interpretable and robust than joint PCA.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import normalize

        # === 1. Sample & aggregate ===
        df_lyric = self._sample_per_author(self.lyrics_df, max_author_songs)
        author_texts = df_lyric.groupby("author")["song_text"].apply(" ".join).reset_index()
        authors = author_texts["author"].tolist()
        n_authors = len(authors)

        if n_authors < 2:
            raise ValueError("Need ≥2 authors.")

        # === 2. TF-IDF ===
        X_tfidf = TfidfVectorizer(max_features=5000).fit_transform(author_texts["song_text"]).toarray()

        # === 3. Transformer ===
        if use_transformer:
            print(f"Computing transformer embeddings for {n_authors} authors...")
            X_trans = np.array([
                self._get_transformer_embeddings(
                    df_lyric[df_lyric["author"] == a]["song_text"].tolist()
                ).mean(axis=0)
                for a in authors
            ])
        else:
            X_trans = X_tfidf

        # === 4. PCA Separately ===
        n_comp = min(n_components, n_authors - 1)

        pca_tfidf = PCA(n_components=n_comp, random_state=42)
        X_tfidf_pca = pca_tfidf.fit_transform(X_tfidf)
        var_tfidf = pca_tfidf.explained_variance_ratio_.sum()

        pca_trans = PCA(n_components=n_comp, random_state=42)
        X_trans_pca = pca_trans.fit_transform(X_trans)
        var_trans = pca_trans.explained_variance_ratio_.sum()

        # === 5. Normalize ===
        X_tfidf_pca = normalize(X_tfidf_pca)
        X_trans_pca = normalize(X_trans_pca)

        # === 6. Cosine similarity ===
        similarities = np.array([
            cosine_similarity([X_tfidf_pca[i]], [X_trans_pca[i]])[0, 0]
            for i in range(n_authors)
        ])

        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)

        # === 7. Print ===
        print(f"\nTF-IDF vs TRANSFORMER (Separate PCA, {n_comp}D each)")
        print(f"TF-IDF variance preserved : {var_tfidf:.1%}")
        print(f"Transformer variance preserved: {var_trans:.1%}")
        print(f"{'Author':<25} {'Cosine'}")
        print("-" * 40)
        for a, s in zip(authors, similarities):
            print(f"{a:<25} {s:.3f}")
        print(f"\nMean cosine = {mean_sim:.3f} ± {std_sim:.3f}")
        if mean_sim > 0.7:
            print("STRONG alignment")
        elif mean_sim > 0.5:
            print("MODERATE alignment")
        else:
            print("WEAK alignment")

        # === 8. Plot ===
        plt.figure(figsize=(10, max(6, n_authors * 0.3)))
        plt.barh(authors, similarities, color='steelblue')
        plt.xlabel("Cosine Similarity (Separate PCA Space)")
        plt.title(f"TF-IDF vs Transformer Alignment\n(Mean = {mean_sim:.3f})")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.show()

        return {
            "authors": authors,
            "similarities": similarities.tolist(),
            "mean": float(mean_sim),
            "tfidf_variance": float(var_tfidf),
            "trans_variance": float(var_trans),
            "n_components": n_comp
        }
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

    def procrustes_alignment(self, max_author_songs=None, use_transformer=False):
        from scipy.linalg import orthogonal_procrustes

        # Content
        df_lyric = self._sample_per_author(self.lyrics_df, max_author_songs)
        if use_transformer:
            Xc = self._get_transformer_embeddings(df_lyric["song_text"].tolist())
        else:
            Xc = TfidfVectorizer(max_features=5000).fit_transform(df_lyric["song_text"]).toarray()
        Xc = PCA(n_components=2).fit_transform(Xc)

        # Style
        df_style = self._sample_per_author(self.styles_df, max_author_songs)
        Xs = self._extract_style_features(df_style)
        Xs = PCA(n_components=2).fit_transform(Xs)

        # Align by author + index
        keys_c = [(a, i) for i, a in enumerate(df_lyric["author"])]
        keys_s = [(a, i) for i, a in enumerate(df_style["author"])]
        common_keys = sorted(set(keys_c) & set(keys_s))
        idx_c = [keys_c.index(k) for k in common_keys]
        idx_s = [keys_s.index(k) for k in common_keys]
        Xc, Xs = Xc[idx_c], Xs[idx_s]
        common_authors = [k[0] for k in common_keys]

        Xc -= Xc.mean(axis=0); Xs -= Xs.mean(axis=0)
        R, sca = orthogonal_procrustes(Xc, Xs)
        Xc_aligned = Xc @ R * sca
        disparity = ((Xc_aligned - Xs) ** 2).sum() / (Xs ** 2).sum()

        unique_authors = np.unique(common_authors)
        cmap = plt.cm.get_cmap("tab20", len(unique_authors))
        color_map = {a: cmap(i) for i, a in enumerate(unique_authors)}

        plt.figure(figsize=(10, 5))
        for auth in unique_authors:
            mask = np.array(common_authors) == auth
            plt.subplot(1, 2, 1).scatter(Xc[mask, 0], Xc[mask, 1], c=[color_map[auth]], s=30, alpha=0.7, label=auth)
            plt.subplot(1, 2, 2).scatter(Xs[mask, 0], Xs[mask, 1], c=[color_map[auth]], s=30, alpha=0.7)
        plt.subplot(1, 2, 1).set_title("Content"); plt.subplot(1, 2, 2).set_title("Style")
        plt.suptitle(f"Procrustes (disparity = {disparity:.4f})")
        plt.tight_layout(); plt.show()
        return disparity

    def plot_similarity_heatmap(self, space='content', max_author_songs=None, cmap='viridis', use_transformer=False):

        if space in ('content', 'both'):
            df_lyric = self._sample_per_author(self.lyrics_df, max_author_songs)
            if use_transformer:
                Xc = self._get_transformer_embeddings(df_lyric["song_text"].tolist())
                df_c = pd.DataFrame(Xc, index=df_lyric["author"])
                Xc = df_c.groupby(level=0).mean().values
            else:
                author_texts = df_lyric.groupby('author')['song_text'].apply(' '.join).reset_index()
                Xc = TfidfVectorizer(max_features=5000).fit_transform(author_texts['song_text']).toarray()

        if space in ('style', 'both'):
            df_style = self._sample_per_author(self.styles_df, max_author_songs)
            Xs = self._extract_style_features(df_style)
            df_s = pd.DataFrame(Xs)
            df_s['author'] = df_style['author'].values
            Xs = df_s.groupby('author').max().values


        if space == 'both':
            authors_c = df_lyric.groupby('author').apply(lambda x: x.name).index.tolist() \
                        if use_transformer else author_texts['author'].tolist()
            authors_s = df_s.groupby('author').max().index.tolist()

            common = sorted(set(authors_c) & set(authors_s))
            idx_c = [authors_c.index(a) for a in common]
            idx_s = [authors_s.index(a) for a in common]

            X = (hstack([Xc[idx_c], Xs[idx_s]]).toarray()
                if not use_transformer else
                np.hstack([Xc[idx_c], Xs[idx_s]]))
        else:
            X = Xc if space == 'content' else Xs
            common = (
                df_lyric.groupby('author').apply(lambda x: x.name).index.tolist()
                if use_transformer else author_texts['author'].tolist()
            ) if space == 'content' else df_s.groupby('author').max().index.tolist()

        sim = cosine_similarity(X)
        n = len(common)

        print("\n=== Per-Author Similarity Rankings ===")

        biggest_diff=1
        for i, author in enumerate(common):
          
            sims = [(common[j], sim[i, j]) for j in range(n) if j != i]
            sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)

            print(f"\n--- {author} ---")
            max_author, max_sim = sims_sorted[0]
            min_author, min_sim = sims_sorted[-1]
            if min_sim<biggest_diff:
                biggest_diff=min_sim
                biggest_diff_author=f'Author {author} has the biggest differnece with {min_author} - {1-min_sim}'
            biggest_drop = max_sim - min_sim

            print(f"\n  Biggest similarity drop: {biggest_drop:.4f}")
            print(f"    From most similar: {max_author} ({max_sim:.4f})")
            print(f"    To least similar:  {min_author} ({min_sim:.4f})")
        print(biggest_diff_author)
        print("\n=============================================\n")

        plt.figure(figsize=(10, 8))
        im = plt.imshow(sim, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(im, label='Cosine similarity')
        plt.xticks(range(len(common)), common, rotation=90, fontsize=8)
        plt.yticks(range(len(common)), common, fontsize=8)
        plt.title(f'Author similarity ({space})')
        plt.tight_layout()
        plt.show()



# === USAGE ===
if __name__ == "__main__":
    viz = SongStyleVisualizer()

    viz.plot_similarity_heatmap(use_transformer=False) 
    # TF-IDF mode
    #viz.distance_rank_correlation(max_author_songs=50, use_transformer=False)

    # TRANSFORMER mode
    #viz.distance_rank_correlation(max_author_songs=50, use_transformer=True)

    #viz.compare_content_vs_style_similarity(top_k=5, max_author_songs=50, use_transformer=True)
    #viz.compare_content_vs_style_similarity(top_k=5, max_author_songs=50, use_transformer=False)
    #viz.plot_all_songs(use_pca=True, max_author_songs=20, use_transformer=True)
    #viz.compare_author_representations()