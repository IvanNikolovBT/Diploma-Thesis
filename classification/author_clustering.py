import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from hdbscan import HDBSCAN
import time
import optuna
import joblib
import warnings
warnings.filterwarnings("ignore")

class Author_Clustering:
    MODEL_NAME = "macedonizer/mk-roberta-base"
    tokenizer = model = device = None

    @classmethod
    def _init_model(cls):
        if cls.tokenizer is None:

            cls.tokenizer = AutoTokenizer.from_pretrained(cls.MODEL_NAME)
            if cls.tokenizer.pad_token is None:
                cls.tokenizer.pad_token = cls.tokenizer.eos_token
            cls.model = AutoModel.from_pretrained(cls.MODEL_NAME)
            cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls.model.to(cls.device).eval()

    def __init__(self, df_path='classification/cleaned_songs.csv', random_state=42):
        self.df = pd.read_csv(df_path)
        self.random_state = random_state
        self._init_model()


    def _get_embeddings(self, texts, mode='bert', batch_size=8):
        if mode == 'bert':
            return self._bert_embeddings(texts, batch_size)
        return self._tfidf_embeddings(texts)

    def _bert_embeddings(self, texts, batch_size=8):
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, truncation=True, 
                                 padding='max_length', return_tensors='pt')
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
                mask = enc['attention_mask'].unsqueeze(-1)
                embs.append((hidden * mask).sum(1) / mask.sum(1).clamp(min=1))
        return torch.cat(embs).cpu().numpy()

    def _tfidf_embeddings(self, texts):
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)
        return vectorizer.fit_transform(texts).toarray()

    def _balance_dataset(self, max_songs):
        np.random.seed(self.random_state)
        balanced = []
        for author in self.df['author'].unique():
            songs = self.df[self.df['author'] == author]
            balanced.append(songs.sample(n=min(len(songs), max_songs), random_state=self.random_state))
        df_bal = pd.concat(balanced).reset_index(drop=True)
        authors = sorted(df_bal['author'].unique())
        a_map = {a: i for i, a in enumerate(authors)}
        r_map = {i: a for a, i in a_map.items()}
        return df_bal, df_bal['author'].map(a_map).values, a_map, r_map


    def _cluster(self, X, algo, n_authors, params=None):
        params = params or {}
        if algo == 'hdbscan':
            return HDBSCAN(min_cluster_size=params.get('min_cluster_size', 5),
                           min_samples=params.get('min_samples', 1)).fit_predict(X)
        if algo == 'ward':
            return AgglomerativeClustering(n_clusters=n_authors, linkage='ward').fit_predict(X)
        if algo == 'gmm':
            return GaussianMixture(n_components=n_authors, random_state=self.random_state).fit_predict(X)
        return KMeans(n_clusters=n_authors, n_init=30, random_state=self.random_state).fit_predict(X)


    def objective(self, trial):
        mode = trial.suggest_categorical('mode', ['bert', 'tfidf'])
        algo = trial.suggest_categorical('cluster_algo', ['kmeans', 'ward', 'gmm', 'hdbscan'])
        max_songs = trial.suggest_int('max_songs', 50, 300, step=50)
        pca_dim = trial.suggest_int('pca_dim', 30, 100, step=10)

        df_sel, y, a_map, r_map = self._balance_dataset(max_songs)
        X_raw = self._get_embeddings(df_sel['song_text'], mode)
        X = StandardScaler().fit_transform(PCA(n_components=min(pca_dim, X_raw.shape[0]-1), random_state=self.random_state).fit_transform(X_raw))

        hdb_params = {}
        if algo == 'hdbscan':
            hdb_params['min_cluster_size'] = trial.suggest_int('min_cluster_size', 3, 15)
            hdb_params['min_samples'] = trial.suggest_int('min_samples', 1, 10)

        start = time.time()
        labels = self._cluster(X, algo, len(a_map), hdb_params)
        elapsed = time.time() - start

        n_clusters = len(set(labels) - {-1}) if -1 in labels else len(set(labels))
        ari = adjusted_rand_score(y, labels) if n_clusters > 1 else 0
        sil = silhouette_score(X, labels) if 1 < n_clusters < len(X) else -1

        trial.set_user_attr("n_clusters", n_clusters)
        return ari


    def optimize(self, n_trials=50):
        study = optuna.create_study(direction="maximize", study_name="poetry_clust", load_if_exists=True)
        study.optimize(self.objective, n_trials=n_trials)
        best = study.best_trial

        print(f"\nBEST: ARI={best.value:.4f} | {best.params}")
        self._run_best(best.params)
        return study

    def _run_best(self, params):
        mode, algo, max_songs, pca_dim = params['mode'], params['cluster_algo'], params['max_songs'], params['pca_dim']
        df_sel, y, a_map, r_map = self._balance_dataset(max_songs)
        X_raw = self._get_embeddings(df_sel['song_text'], mode)
        X = StandardScaler().fit_transform(PCA(n_components=min(pca_dim, X_raw.shape[0]-1), random_state=self.random_state).fit_transform(X_raw))
        hdb_params = {k: params[k] for k in params if k.startswith('min_')}
        labels = self._cluster(X, algo, len(a_map), hdb_params)
        joblib.dump({'X': X, 'y': y, 'labels': labels, 'a_map': a_map, 'r_map': r_map, 'params': params}, 'best_model.pkl')



if __name__ == "__main__":
    exp = Author_Clustering()
    exp.optimize(n_trials=1)  