import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
import seaborn as sns


class AuthorClassifier:
    def __init__(self, max_per_author=300, min_author_samples=10, test_size=0.2, batch_size=32):
        self.max_per_author = max_per_author
        self.min_author_samples = min_author_samples
        self.test_size = test_size
        self.batch_size = batch_size


        self.best_params = {
            'max_features': 4619,
            'n_layers': 1,
            'neurons': 567,
            'activation': 'tanh',
            'dropout_rate': 0.3406819279083615,
            'optimizer': 'rmsprop',
            'lr': 0.0007878787378953067,
            'l2_reg': 3.145848564707723e-05,
            'n_epochs': 41,
            'min_df': 3,
            'max_df': 0.8904674508605334,
            'ngram_range': (1, 1)
        }

        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.history = None

    def _limit_per_author(self, df, max_per_author):
        return df.groupby("author", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max_per_author), random_state=42)
        )

    def fit(self, train_csv_path, verbose=0):
        print("Читање на тренинг податоци...")
        df_authentic = pd.read_csv(train_csv_path)
        df_authentic['song_text'] = df_authentic['song_text'].str.lower()

        df_limited = self._limit_per_author(df_authentic, self.max_per_author)
        print(f"Песни по тренирање (лимитирани): {len(df_limited)}")

        author_counts = df_limited['author'].value_counts()
        valid_authors = author_counts[author_counts >= self.min_author_samples].index
        df_filtered = df_limited[df_limited['author'].isin(valid_authors)]
        print(f"Останати автори: {len(valid_authors)}, песни: {len(df_filtered)}")

        print("Векторизација со TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.best_params['max_features'],
            ngram_range=self.best_params['ngram_range'],
            min_df=self.best_params['min_df'],
            max_df=self.best_params['max_df']
        )
        X = self.vectorizer.fit_transform(df_filtered['song_text']).toarray()

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df_filtered['author'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )

        print("Градење на невронска мрежа...")
        input_layer = layers.Input(shape=(X_train.shape[1],))
        x = input_layer
        for _ in range(self.best_params['n_layers']):
            x = layers.Dense(
                self.best_params['neurons'],
                activation=self.best_params['activation'],
                kernel_regularizer=regularizers.l2(self.best_params['l2_reg'])
            )(x)
            x = layers.Dropout(self.best_params['dropout_rate'])(x)
        output = layers.Dense(len(self.label_encoder.classes_), activation="softmax")(x)

        self.model = keras.Model(inputs=input_layer, outputs=output)
        optimizer = keras.optimizers.RMSprop(learning_rate=self.best_params['lr'])
        self.model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

 
        print(f"Тренирање за {self.best_params['n_epochs']} епохи...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=self.best_params['n_epochs'],
            batch_size=self.batch_size,
            verbose=verbose
        ).history

        print("Евалуација на тренинг сетот...")
        y_pred = np.argmax(self.model.predict(X_test, verbose=0), axis=1)
        print("\n=== Author Prediction (Test Set) ===")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_, zero_division=0))
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"Weighted F1: {f1:.4f}")

        cm = tf.math.confusion_matrix(y_test, y_pred).numpy()
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        plt.title('Author Prediction - Confusion Matrix (Test Set)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()

        return self

    def evaluate_generated(self, eval_csv_path):
        if self.model is None or self.vectorizer is None:
            raise ValueError("Моделот не е трениран! Прво повикај .fit()")

        print("Читање на генерирани песни...")
        df_gen = pd.read_csv(eval_csv_path)
        df_gen['new_song'] = df_gen['new_song'].str.lower()
        df_gen.rename(columns={'new_song': 'song_text'}, inplace=True)

        df_limited = self._limit_per_author(df_gen, self.max_per_author)
        print(f"Генерирани песни (лимитирани): {len(df_limited)}")


        X_gen = self.vectorizer.transform(df_limited['song_text']).toarray()
        y_true = self.label_encoder.transform(df_limited['author'])

        print("Предвидување на генерирани песни...")
        y_pred = np.argmax(self.model.predict(X_gen, verbose=0), axis=1)

        print("\n=== Author Prediction on Generated Songs ===")
        print(classification_report(y_true, y_pred, target_names=self.label_encoder.classes_, zero_division=0))
        f1 = f1_score(y_true, y_pred, average="weighted")
        print(f"Weighted F1 (Generated): {f1:.4f}")

        cm = tf.math.confusion_matrix(y_true, y_pred).numpy()
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        plt.title('Author Prediction - Generated Songs')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()

        return f1


    def plot_history(self):
        if self.history is None:
            print("Нема историја! Прво тренирај.")
            return

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['accuracy'], label='Train Acc')
        plt.plot(self.history['val_accuracy'], label='Val Acc')
        plt.title('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()