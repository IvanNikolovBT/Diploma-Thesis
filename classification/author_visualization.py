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
from scipy.stats import spearmanr
from scipy.linalg import orthogonal_procrustes
from scipy.linalg import orthogonal_procrustes
class SongStyleVisualizer:
    # Default paths — can be overridden in __init__
    DEFAULT_CSV_PATH = "/home/ivan/Desktop/Diplomska/final_results_csv/cleaned_songs_with_perplexity.csv"
    DEFAULT_STYLE_CSV_PATH = "/home/ivan/Desktop/Diplomska/api_styles_all_in_one_text.csv"

    def __init__(self, csv_path=None, style_csv_path=None):
        """
        Load both CSVs once at initialization.
        """
        self.csv_path = csv_path or self.DEFAULT_CSV_PATH
        self.style_csv_path = style_csv_path or self.DEFAULT_STYLE_CSV_PATH

        print(f"Loading lyrics from: {self.csv_path}")
        self.lyrics_df = pd.read_csv(self.csv_path)
        self.lyrics_df["song_text"] = self.lyrics_df["song_text"].fillna("").astype(str)

        print(f"Loading styles from: {self.style_csv_path}")
        self.styles_df = pd.read_csv(self.style_csv_path)
        self.styles_df["extracted_styles"] = self.styles_df["extracted_styles"].fillna("").astype(str)

        self.authors = None  # Will be set per method

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
        reducer = PCA(n_components=2) if use_pca else TSNE(n_components=2, perplexity=20, learning_rate=100, init="random")
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

    def plot_all_songs(self, use_pca=False, max_author_songs=None, label_clusters=False):
        df = self._sample_per_author(self.lyrics_df, max_author_songs)
        self.authors = df["author"].unique()

        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf = vectorizer.fit_transform(df["song_text"])
        reduced = self._reduce_dimensions(tfidf, use_pca)

        xcol, ycol = ("pca_x", "pca_y") if use_pca else ("tsne_x", "tsne_y")
        df[xcol], df[ycol] = reduced[:, 0], reduced[:, 1]
        title = f"TF-IDF Songs Projected to 2D ({'PCA' if use_pca else 't-SNE'})"

        plt.figure(figsize=(12, 8))
        for author in self.authors:
            mask = df["author"] == author
            plt.scatter(df.loc[mask, xcol], df.loc[mask, ycol], label=author, s=25, alpha=0.7)

        self._add_labels_with_clusters(df, xcol, ycol, label_clusters, eps=1.0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(title); plt.xlabel(xcol); plt.ylabel(ycol)
        plt.tight_layout(); plt.show()

    def plot_authors_aggregate(self, use_pca=False):
        df = self.lyrics_df.copy()
        author_texts = df.groupby("author")["song_text"].apply(" ".join).reset_index()
        tfidf = TfidfVectorizer(max_features=5000).fit_transform(author_texts["song_text"])
        reduced = self._reduce_dimensions(tfidf, use_pca)

        author_texts[["x", "y"]] = reduced
        title = f"Authors TF-IDF Projected to 2D ({'PCA' if use_pca else 't-SNE'})"

        plt.figure(figsize=(12, 8))
        plt.scatter(author_texts["x"], author_texts["y"], s=60, alpha=0.7)
        for _, row in author_texts.iterrows():
            plt.text(row["x"], row["y"], row["author"], fontsize=9, weight="bold",
                     ha="center", va="center", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5))
        plt.title(title); plt.xlabel("Component 1"); plt.ylabel("Component 2")
        plt.tight_layout(); plt.show()

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

    def plot_author_styles(self, use_pca=False, max_author_songs=None, label_clusters=False):
        df = self._sample_per_author(self.styles_df, max_author_songs)
        self.authors = df["author"].unique()

        features = self._extract_style_features(df)
        reduced = self._reduce_dimensions(features, use_pca)

        xcol, ycol = ("pca_x", "pca_y") if use_pca else ("tsne_x", "tsne_y")
        df[xcol], df[ycol] = reduced[:, 0], reduced[:, 1]
        title = f"Author Styles Projected to 2D ({'PCA' if use_pca else 't-SNE'})"

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
        title = f"Aggregated Author Styles Projected to 2D ({'PCA' if use_pca else 't-SNE'})"

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

    def compare_content_vs_style_similarity(self, top_k=5, max_author_songs=None):
        # Sample both datasets
        lyrics_sample = self._sample_per_author(self.lyrics_df, max_author_songs)
        styles_sample = self._sample_per_author(self.styles_df, max_author_songs)

        # === CONTENT SPACE ===
        author_texts = lyrics_sample.groupby("author")["song_text"].apply(" ".join).reset_index()
        vectorizer = TfidfVectorizer(max_features=5000)
        content_matrix = vectorizer.fit_transform(author_texts["song_text"])
        content_sim = cosine_similarity(content_matrix)

        # === STYLE SPACE ===
        style_features = self._extract_style_features(styles_sample)
        df_style = pd.DataFrame(style_features)
        df_style["author"] = styles_sample["author"].values
        style_agg = df_style.groupby("author").max().values
        style_sim = cosine_similarity(style_agg)

        # Align authors
        authors = author_texts["author"].tolist()
        author_to_idx = {auth: i for i, auth in enumerate(authors)}

        def get_neighbors(sim_matrix, idx, k):
            scores = sim_matrix[idx].copy()
            scores[idx] = -1
            neighbors_idx = np.argsort(scores)[-k:][::-1]
            return [authors[i] for i in neighbors_idx]

        results = {}
        overlaps = []
        for author in authors:
            idx = author_to_idx[author]
            content_neighbors = get_neighbors(content_sim, idx, top_k)
            style_neighbors = get_neighbors(style_sim, idx, top_k)
            overlap = len(set(content_neighbors) & set(style_neighbors))
            jaccard = overlap / (2 * top_k)
            results[author] = {
                "content_neighbors": content_neighbors,
                "style_neighbors": style_neighbors,
                "overlap_count": overlap,
                "jaccard_alignment": jaccard
            }
            overlaps.append(jaccard)

        mean_alignment = np.mean(overlaps)
        std_alignment = np.std(overlaps)

        print(f"\nCONTENT vs STYLE SIMILARITY ALIGNMENT (Top-{top_k} NN)")
        print(f"{'Author':<25} {'Overlap':<8} {'Jaccard':<8} Content NN → Style NN")
        print("-" * 80)
        for author in sorted(results, key=lambda x: results[x]["jaccard_alignment"], reverse=True):
            r = results[author]
            print(f"{author:<25} {r['overlap_count']:<8} {r['jaccard_alignment']:.3f}    "
                  f"{', '.join(r['content_neighbors'][:3])}{'...' if len(r['content_neighbors'])>3 else ''} → "
                  f"{', '.join(r['style_neighbors'][:3])}{'...' if len(r['style_neighbors'])>3 else ''}")

        print(f"\nGLOBAL: Mean Jaccard = {mean_alignment:.3f} ± {std_alignment:.3f}")
        if mean_alignment > 0.4:
            print("STRONG alignment: Similar writing → similar extracted style")
        elif mean_alignment > 0.2:
            print("MODERATE alignment")
        else:
            print("WEAK alignment: Style extraction may not reflect lyrical similarity")


        return results, mean_alignment
    def plot_similarity_heatmap(self, space='content', max_author_songs=None, cmap='viridis'):
        if space in ('content', 'both'):
            df_lyric = self._sample_per_author(self.lyrics_df, max_author_songs)
            author_texts = df_lyric.groupby('author')['song_text'].apply(' '.join).reset_index()
            vec = TfidfVectorizer(max_features=5000)
            Xc = vec.fit_transform(author_texts['song_text'])
        if space in ('style', 'both'):
            df_style = self._sample_per_author(self.styles_df, max_author_songs)
            Xs = self._extract_style_features(df_style)
            df_s = pd.DataFrame(Xs, columns=[f'style_{i}' for i in range(Xs.shape[1])])
            df_s['author'] = df_style['author'].values
            Xs = df_s.groupby('author').max().values

        
        if space == 'both':
            authors_c = author_texts['author'].tolist()
            authors_s = df_s.groupby('author').max().index.tolist()
            common = sorted(set(authors_c) & set(authors_s))
            Xc = Xc[[authors_c.index(a) for a in common]]
            Xs = Xs[[authors_s.index(a) for a in common]]
            from scipy.sparse import hstack
            X = hstack([Xc, Xs]).toarray()
        else:
            X = Xc if space == 'content' else Xs
            common = author_texts['author'].tolist() if space == 'content' else df_s.groupby('author').max().index.tolist()

       
        sim = cosine_similarity(X)

        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(sim, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(im, label='Cosine similarity')
        plt.xticks(range(len(common)), common, rotation=90, fontsize=8)
        plt.yticks(range(len(common)), common, fontsize=8)
        plt.title(f'Author similarity ({space})')
        plt.tight_layout()
        plt.show()
    def get_distances(self,X):
        sim=cosine_similarity(X)
        n=sim.shape[0]
        dists=[]
        for i in range(n):
            for j in range(i+1,n):
                dists.append(1-sim[i,j])
        return np.array(dists)
    def distance_rank_correlation(self, max_author_songs=None):
        
        df_lyric = self._sample_per_author(self.lyrics_df, max_author_songs)
        txt = df_lyric.groupby('author')['song_text'].apply(' '.join).reset_index()
        Xc = TfidfVectorizer(max_features=5000).fit_transform(txt['song_text']).toarray()
        dc = self.get_distances(Xc)

        
        df_style = self._sample_per_author(self.styles_df, max_author_songs)
        Xs = self._extract_style_features(df_style)
        df_s = pd.DataFrame(Xs); df_s['author'] = df_style['author'].values
        Xs = df_s.groupby('author').max().values
        ds = self.get_distances(Xs)

        rho, p = spearmanr(dc, ds)
        print(f"Spearman ρ = {rho:.3f} (p = {p:.2e})")
        if rho > 0.5:   print("Strong rank agreement")
        elif rho > 0.2: print("Moderate")
        else:           print("Weak / none")

        return rho, p
   
    
viz = SongStyleVisualizer() 
viz.distance_rank_correlation(max_author_songs=50)


