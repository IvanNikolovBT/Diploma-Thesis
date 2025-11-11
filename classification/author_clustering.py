# Macedonian Poetry Experiments: Unified Class with Logging
# Modes: BERT or TF-IDF | Algos: KMeans, HDBSCAN, Ward, GMM
# Auto-logs params, results, time to 'poetry_experiments.log'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE
from hdbscan import HDBSCAN
import time
import logging
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    filename='poetry_experiments.log',
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class UnifiedPoetryExperimenter:
    # === CONFIG ===
    MODEL_NAME = "macedonizer/mk-roberta-base"
    tokenizer = None
    model = None
    device = None

    @classmethod
    def _init_transformer(cls):
        if cls.tokenizer is None:
            logging.info("Loading macedonizer/mk-roberta-base...")
            print("Loading macedonizer/mk-roberta-base...")
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.MODEL_NAME)
            if cls.tokenizer.pad_token is None:
                cls.tokenizer.pad_token = cls.tokenizer.eos_token
            cls.model = AutoModel.from_pretrained(cls.MODEL_NAME)
            cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {cls.device}")
            print(f"Using device: {cls.device}")
            cls.model.to(cls.device)
            cls.model.eval()

    def __init__(self, configs, batch_size=8, max_len=512, random_state=42, plot=True):
        """
        configs: List of dicts, each with:
            - 'mode': 'bert' or 'tfidf'
            - 'cluster_algo': 'kmeans', 'hdbscan', 'ward', 'gmm'
            - 'max_songs': int (e.g., 10, 100, 300)
        """
        self.df = pd.read_csv('classification/cleaned_songs.csv')
        self.configs = configs
        self.batch_size = batch_size
        self.max_len = max_len
        self.random_state = random_state
        self.plot = plot
        self.results = []  # Store all experiment results

        # Load BERT if any config needs it
        if any(c['mode'] == 'bert' for c in configs):
            self._init_transformer()

    # === EMBEDDING METHODS ===
    def _get_bert_embeddings(self, texts):
        all_emb = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tokenizer(
                batch, truncation=True, max_length=self.max_len,
                padding='max_length', return_tensors="pt"
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

    def _get_tfidf_embeddings(self, texts):
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            lowercase=True,
            sublinear_tf=True,
            stop_words=None
        )
        return vectorizer.fit_transform(texts).toarray()

    # === DATA PREP ===
    def _balance_dataset(self, max_songs):
        balanced = []
        np.random.seed(self.random_state)
        for author in self.df['author'].unique():
            auth_songs = self.df[self.df['author'] == author]
            n_take = min(len(auth_songs), max_songs)
            auth_songs = auth_songs.sample(n=n_take, random_state=self.random_state)
            balanced.append(auth_songs)
        selected_df = pd.concat(balanced).reset_index(drop=True)
        authors = sorted(selected_df['author'].unique())
        author_map = {auth: idx for idx, auth in enumerate(authors)}
        reverse_map = {idx: auth for auth, idx in author_map.items()}
        y = selected_df['author'].map(author_map).values
        return selected_df, y, author_map, reverse_map

    def _prepare_features(self, texts, mode):
        print(f"Extracting features with {mode.upper()}...")
        if mode == 'bert':
            raw_emb = self._get_bert_embeddings(texts.tolist())
        else:
            raw_emb = self._get_tfidf_embeddings(texts)
        
        n_comp = min(50, raw_emb.shape[0] - 1, raw_emb.shape[1])
        pca = PCA(n_components=n_comp, random_state=self.random_state)
        X_pca = pca.fit_transform(raw_emb)
        X = StandardScaler().fit_transform(X_pca)
        
        print(f"  Raw: {raw_emb.shape} → PCA({n_comp}D)")
        return X

    # === CLUSTERING ===
    def _cluster(self, X, y, cluster_algo, author_map, reverse_map):
        print(f"Clustering with {cluster_algo.upper()}...")
        n_authors = len(author_map)
        
        if cluster_algo == 'hdbscan':
            clusterer = HDBSCAN(min_cluster_size=5, min_samples=3)
            labels = clusterer.fit_predict(X)
            n_clusters = len(set(labels) - {-1})
            noise = np.sum(labels == -1)
            print(f"Found {n_clusters} clusters + {noise} noise")

        elif cluster_algo == 'ward':
            clusterer = AgglomerativeClustering(n_clusters=n_authors, linkage='ward')
            labels = clusterer.fit_predict(X)
            n_clusters = n_authors

        elif cluster_algo == 'gmm':
            gmm = GaussianMixture(n_components=n_authors, random_state=self.random_state)
            labels = gmm.fit_predict(X)
            n_clusters = n_authors

        else:  # kmeans
            kmeans = KMeans(n_clusters=n_authors, n_init=30, random_state=self.random_state)
            labels = kmeans.fit_predict(X)
            n_clusters = n_authors

        # Metrics
        ari = adjusted_rand_score(y, labels) if len(set(labels)) > 1 else 0
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 and len(set(labels)) < len(X) else 0

        print(f"ARI: {ari:.3f} | Silhouette: {sil:.3f}")

        # Purity
        print("\nCluster → Author (Purity):")
        for cluster_id in sorted(np.unique(labels)):
            if cluster_id == -1:
                print(f"  Noise → Outliers | {np.sum(labels == -1)} points")
                continue
            mask = labels == cluster_id
            true_authors = y[mask]
            if len(true_authors) == 0: continue
            dominant = Counter(true_authors).most_common(1)[0]
            auth_id, count = dominant
            total = mask.sum()
            purity = count / total
            print(f"  Cluster {cluster_id:2d} → {reverse_map[auth_id]:25s} | {purity:6.1%} pure | {count}/{total}")

        return labels, ari, sil, n_clusters

    # === VISUALIZATION ===
    def _visualize_2d(self, X, y, labels, mode, cluster_algo, max_songs, reverse_map):
        perplexity = min(30, max(5, len(X)//5))
        tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=perplexity)
        X_2d = tsne.fit_transform(X)

        plt.figure(figsize=(14, 10))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab20', s=60, alpha=0.85, edgecolors='k', linewidth=0.3)
        plt.colorbar(scatter, label='Cluster ID', shrink=0.8)
        plt.title(f"t-SNE: Macedonian Poetry ({mode.upper()} + {cluster_algo.upper()})\n"
                  f"Max {max_songs} songs/author | ARI: {adjusted_rand_score(y, labels):.3f}", fontsize=14)

        for auth_id, auth_name in reverse_map.items():
            idxs = np.where(y == auth_id)[0]
            if len(idxs) > 0:
                center_idx = idxs[len(idxs)//2]
                plt.annotate(auth_name, (X_2d[center_idx, 0], X_2d[center_idx, 1]),
                             fontsize=8, ha='center',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"plot_{mode}_{cluster_algo}_{max_songs}.png")
        plt.show()

    # === RUN ALL EXPERIMENTS ===
    def run_all(self):
        for config in self.configs:
            mode = config['mode']
            cluster_algo = config['cluster_algo']
            max_songs = config['max_songs']
            
            start_time = time.time()
            
            # Prep data
            selected_df, y, author_map, reverse_map = self._balance_dataset(max_songs)
            texts = selected_df['song_text'].fillna('').astype(str)
            
            # Features
            X = self._prepare_features(texts, mode)
            
            # Cluster
            labels, ari, sil, n_clusters = self._cluster(X, y, cluster_algo, author_map, reverse_map)
            
            time_taken = time.time() - start_time
            
            # Visualize if enabled
            if self.plot:
                self._visualize_2d(X, y, labels, mode, cluster_algo, max_songs, reverse_map)
            
            # Log
            log_msg = (
                f"EXPERIMENT | Mode: {mode.upper()} | Algo: {cluster_algo.upper()} | "
                f"Max Songs: {max_songs} | Authors: {len(author_map)} | Songs: {len(selected_df)} | "
                f"Clusters Found: {n_clusters} | ARI: {ari:.3f} | Silhouette: {sil:.3f} | "
                f"Time: {time_taken:.2f}s"
            )
            logging.info(log_msg)
            print(log_msg)
            
            # Store result
            self.results.append({
                'config': config,
                'ari': ari,
                'sil': sil,
                'time': time_taken,
                'labels': labels,
                'X': X,
                'y': y
            })



configs = [
    {'mode': 'bert', 'cluster_algo': 'kmeans', 'max_songs': 10},
    {'mode': 'bert', 'cluster_algo': 'kmeans', 'max_songs': 100},
    {'mode': 'bert', 'cluster_algo': 'kmeans', 'max_songs': 300},
    {'mode': 'tfidf', 'cluster_algo': 'kmeans', 'max_songs': 10},
    {'mode': 'tfidf', 'cluster_algo': 'kmeans', 'max_songs': 100},
    {'mode': 'tfidf', 'cluster_algo': 'kmeans', 'max_songs': 300},
    {'mode': 'bert', 'cluster_algo': 'hdbscan', 'max_songs': 300},
    {'mode': 'tfidf', 'cluster_algo': 'hdbscan', 'max_songs': 300},
    
]

experimenter = UnifiedPoetryExperimenter(configs, plot=True)
experimenter.run_all()

# View log: cat poetry_experiments.log
#10
# BERT  → ARI: 0.083 | Silhouette: 0.007 
#TF-IDF → ARI: 0.122 | Silhouette: 0.036 

#100
# BERT  → ARI: 0.155 | Silhouette: 0.007 
#TF-IDF → ARI: 0.125 | Silhouette: 0.054 

#300
#BERT  → ARI: 0.174 | Silhouette: 0.002
#TF-IDF → ARI: 0.084 | Silhouette: 0.036