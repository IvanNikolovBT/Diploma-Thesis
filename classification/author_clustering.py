
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
from scipy.cluster.hierarchy import dendrogram, linkage
import time
import logging
import optuna
import joblib
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    filename='poetry_experiments.log',
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)


class UnifiedPoetryExperimenter:
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

    def __init__(self, df_path='classification/cleaned_songs.csv', random_state=42):
        self.df = pd.read_csv(df_path)
        self.random_state = random_state
        self._init_transformer()


    def _get_bert_embeddings(self, texts, batch_size=8, max_len=512):
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch, truncation=True, max_length=max_len,
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
            max_features=5000, ngram_range=(1, 2), lowercase=True,
            sublinear_tf=True, stop_words=None
        )
        return vectorizer.fit_transform(texts).toarray()


    def _balance_dataset(self, max_songs):
        balanced = []
        np.random.seed(self.random_state)
        for author in self.df['author'].unique():
            auth_songs = self.df[self.df['author'] == author]
            n_take = min(len(auth_songs), max_songs)
            sampled = auth_songs.sample(n=n_take, random_state=self.random_state)
            balanced.append(sampled)
        df_balanced = pd.concat(balanced).reset_index(drop=True)
        authors = sorted(df_balanced['author'].unique())
        author_map = {a: i for i, a in enumerate(authors)}
        reverse_map = {i: a for a, i in author_map.items()}
        y = df_balanced['author'].map(author_map).values
        return df_balanced, y, author_map, reverse_map


    def objective(self, trial):
        mode        = trial.suggest_categorical('mode', ['bert', 'tfidf'])
        cluster_algo= trial.suggest_categorical('cluster_algo', ['kmeans', 'hdbscan', 'ward', 'gmm'])
        max_songs   = trial.suggest_int('max_songs', 50, 300, step=50)
        pca_dim     = trial.suggest_int('pca_dim', 30, 100, step=10)


        hdb_params = {}
        if cluster_algo == 'hdbscan':
            hdb_params['min_cluster_size'] = trial.suggest_int('min_cluster_size', 3, 15)
            hdb_params['min_samples']      = trial.suggest_int('min_samples', 1, 10)

        start_time = time.time()


        df_sel, y, a_map, r_map = self._balance_dataset(max_songs)
        texts = df_sel['song_text'].fillna('').astype(str)


        if mode == 'bert':
            raw = self._get_bert_embeddings(texts.tolist(), batch_size=8)
        else:
            raw = self._get_tfidf_embeddings(texts)

        n_comp = min(pca_dim, raw.shape[0] - 1, raw.shape[1])
        X = StandardScaler().fit_transform(
                PCA(n_components=n_comp, random_state=self.random_state).fit_transform(raw)
            )

        if cluster_algo == 'hdbscan':
            clusterer = HDBSCAN(**hdb_params, metric='euclidean')
            labels = clusterer.fit_predict(X)
        elif cluster_algo == 'ward':
            clusterer = AgglomerativeClustering(n_clusters=len(a_map), linkage='ward')
            labels = clusterer.fit_predict(X)
        elif cluster_algo == 'gmm':
            gmm = GaussianMixture(n_components=len(a_map), random_state=self.random_state)
            labels = gmm.fit_predict(X)
        else:  
            kmeans = KMeans(n_clusters=len(a_map), n_init=30, random_state=self.random_state)
            labels = kmeans.fit_predict(X)

        n_clusters = len(set(labels) - {-1}) if -1 in labels else len(set(labels))
        ari = adjusted_rand_score(y, labels) if n_clusters > 1 else 0
        sil = silhouette_score(X, labels) if n_clusters > 1 and n_clusters < len(X) else -1

        elapsed = time.time() - start_time


        log_msg = (
            f"TRIAL {trial.number:03d} | Mode:{mode.upper()} | Algo:{cluster_algo.upper()} | "
            f"MaxSongs:{max_songs} | PCA:{n_comp} | Authors:{len(a_map)} | Songs:{len(df_sel)} | "
            f"Clusters:{n_clusters} | ARI:{ari:.4f} | Sil:{sil:.4f} | Time:{elapsed:.2f}s"
        )
        if cluster_algo == 'hdbscan':
            log_msg += f" | HDBSCAN(c={hdb_params['min_cluster_size']},s={hdb_params['min_samples']})"
        logging.info(log_msg)
        print(log_msg)

        trial.set_user_attr("n_clusters", n_clusters)
        trial.set_user_attr("time", elapsed)

        return ari


    def optimize(self, n_trials=100, study_name="optuna_poetry_clustering", storage="sqlite:///optuna_poetry_clustering.db"):
        print(f"\n=== OPTUNA START | {n_trials} trials ===")
        logging.info(f"OPTUNA START | n_trials={n_trials}")

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            load_if_exists=True          
        )
        study.optimize(self.objective, n_trials=n_trials,n_jobs=3)

        best = study.best_trial
        print("\nBEST RESULT")
        print(f"  ARI : {best.value:.4f}")
        print(f"  Params: {best.params}")
        logging.info(f"BEST ARI:{best.value:.4f} | Params:{best.params}")

        self._run_and_save_best(best.params)
        return study


    def _run_and_save_best(self, params):
        mode, algo, max_songs, pca_dim = (
            params['mode'], params['cluster_algo'], params['max_songs'], params['pca_dim']
        )
        logging.info("RUNNING BEST CONFIGURATION")
        print("Running best configuration...")

        df_sel, y, a_map, r_map = self._balance_dataset(max_songs)
        texts = df_sel['song_text'].fillna('').astype(str)

        raw = (self._get_bert_embeddings(texts.tolist()) if mode == 'bert'
               else self._get_tfidf_embeddings(texts))

        n_comp = min(pca_dim, raw.shape[0]-1, raw.shape[1])
        X = StandardScaler().fit_transform(
                PCA(n_components=n_comp, random_state=self.random_state).fit_transform(raw)
            )


        if algo == 'hdbscan':
            labels = HDBSCAN(
                min_cluster_size=params.get('min_cluster_size'),
                min_samples=params.get('min_samples')
            ).fit_predict(X)
        elif algo == 'ward':
            labels = AgglomerativeClustering(n_clusters=len(a_map), linkage='ward').fit_predict(X)
        elif algo == 'gmm':
            labels = GaussianMixture(n_components=len(a_map), random_state=self.random_state).fit_predict(X)
        else:  
            labels = KMeans(n_clusters=len(a_map), n_init=30, random_state=self.random_state).fit_predict(X)

        ari = adjusted_rand_score(y, labels)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0


        joblib.dump({
            'X': X, 'y': y, 'labels': labels,
            'author_map': a_map, 'reverse_map': r_map,
            'params': params, 'ari': ari, 'sil': sil
        }, 'best_poetry_model.pkl')


        self._plot_tsne(X, y, labels, mode, algo, max_songs, r_map, suffix="_BEST")
        if algo == 'ward':
            self._plot_dendrogram(X, y, r_map)

        logging.info(f"BEST MODEL SAVED | ARI:{ari:.4f}")
        print("Best model saved -> best_poetry_model.pkl")


    def _plot_tsne(self, X, y, labels, mode, algo, max_songs, r_map, suffix=""):
        perp = min(30, max(5, len(X)//5))
        X2d = TSNE(n_components=2, random_state=self.random_state, perplexity=perp).fit_transform(X)
        plt.figure(figsize=(14,10))
        plt.scatter(X2d[:,0], X2d[:,1], c=labels, cmap='tab20', s=60, alpha=.85, edgecolor='k', linewidth=.3)
        plt.colorbar(label='Cluster')
        plt.title(f"{mode.upper()} + {algo.upper()} | Max {max_songs} | ARI {adjusted_rand_score(y,labels):.3f}")
        for aid, name in r_map.items():
            idx = np.where(y == aid)[0]
            if len(idx): plt.annotate(name, (X2d[idx[len(idx)//2],0], X2d[idx[len(idx)//2],1]),
                                     fontsize=8, ha='center',
                                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        plt.tight_layout()
        plt.savefig(f"plot{suffix}.png", dpi=150)
        plt.show()

    def _plot_dendrogram(self, X, y, r_map):
        Z = linkage(X, 'ward')
        plt.figure(figsize=(14,8))
        dendrogram(Z, labels=[r_map[i] for i in y], leaf_rotation=90,
                   truncate_mode='level', p=6)
        plt.title("Macedonian Poetry – Ward Hierarchy")
        plt.xlabel("Authors")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.savefig("dendrogram_best.png", dpi=150)
        plt.show()



if __name__ == "__main__":
    exp = UnifiedPoetryExperimenter()
    study = exp.optimize(n_trials=100)   
    """"[I 2025-11-11 13:22:54,446] Trial 111 finished with value: 0.1882867578267587 and parameters: {'mode': 'bert', 'cluster_algo': 'kmeans', 'max_songs': 50, 'pca_dim': 40}. Best is trial 20 with value: 0.22013754522382267.

BEST RESULT
  ARI : 0.2201
  Params: {'mode': 'bert', 'cluster_algo': 'ward', 'max_songs': 50, 'pca_dim': 30}
BEST ARI:0.2201 | Params:{'mode': 'bert', 'cluster_algo': 'ward', 'max_songs': 50, 'pca_dim': 30}
RUNNING BEST CONFIGURATION
Running best configuration...
BEST MODEL SAVED | ARI:0.2201
Best model saved -> best_poetry_model.pkl"""