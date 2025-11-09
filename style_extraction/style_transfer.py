import sys
import os
import csv
import pandas as pd
import random
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from poetry_DB import PoetryDB
import time
from tqdm import tqdm
import boto3
import json
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import spacy
from RAG_builder.vectorbuilder import VectorDBBuilder
class StyleTransfer:
    
    def __init__(self):

        self.db=PoetryDB()
        self.CSV_PATH="final_results_csv/cleaned_songs_with_perplexity.csv"
        self.df=pd.read_csv(self.CSV_PATH)
        self.random_seed=47
        self.styles=self.load_styles()
        self.client = boto3.client("bedrock-runtime",region_name="eu-central-1",)
        self.vector_db=VectorDBBuilder()
        self.spacy=spacy.load("mk_core_news_lg")
    
    def load_styles(self):
        abs_path = os.path.abspath('style_extraction/styles.json')
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"❌ Styles file not found: {abs_path}")
        with open(abs_path, "r", encoding="utf-8") as f:
            return json.load(f)  
    def extract_n_random_songs_for_author(self, author_name='Блаже Конески', number_of_songs=10):
        author_songs = self.df[self.df['author'] == author_name]
        return author_songs.sample(
            n=min(number_of_songs, len(author_songs)),
            random_state=None  
        )
    def extract_n_random_songs_with_styles_for_author(self, author_name='Блаже Конески', number_of_songs=10):
        styles=pd.read_csv('api_styles_all_in_one_text.csv')
        author_songs = styles[styles['author'] == author_name]
        return author_songs.sample(
            n=min(number_of_songs, len(author_songs)),
            random_state=None  
        )
    def invoke_nova_micro(self, prompt, system):
        response = self.client.converse(
            modelId="arn:aws:bedrock:eu-central-1::inference-profile/eu.amazon.nova-micro-v1:0", 
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            system=[{"text": system}],
            inferenceConfig={
                "maxTokens": 1786,
                "temperature": 0.5,
                "topP": 0.9
            }
        )
        return response
    def extract_styles_from_song_using_api(self,song,author,song_title,output_path,without_def=True):
        system="Ти си разговорник за екстракција на стил на македонска поезија."
        if without_def:
            prompt=self.create_full_extraction_prompt_without_definition(song)
        else:
            prompt=self.create_full_extraction_prompt_with_definition(song)
        result=self.invoke_nova_micro(prompt,system)
        self.write_to_csv(author,song_title,result,output_path) 
    def extract_all_styles_api(self):
        output_path = 'api_styles_all_in_one_text.csv'

        if os.path.exists(output_path):
            result_df = pd.read_csv(output_path)
            print(f"Loaded existing CSV with {len(result_df)} rows.")
        else:
            cols = ['author', 'song_title', 'extracted_styles',
                    'input_tokens', 'output_tokens', 'total_tokens', 'ms']
            result_df = pd.DataFrame(columns=cols)
            result_df.to_csv(output_path, index=False)
            print("Created new CSV file.")

        i=0
        n=len(self.df)
        for _, song_row in self.df.iterrows():
            author = song_row['author']
            song_title = song_row['song_title']
            song=song_row['song_text']
            exists = (
                (result_df['author'] == author) &
                (result_df['song_title'] == song_title)
            ).any()

            if exists:
                print(f"Skipping {author} - {song_title} (already processed).")
                continue

            print(f"Processing {author} - {song_title}... {i}/{n} {i/n}")
            
            
            self.extract_styles_from_song_using_api(song=song,author=author,
                                                             song_title=song_title,output_path=output_path)


        print("✅ All songs processed.")
    def create_full_extraction_prompt_without_definition(self,song: str) -> str:

        features_text = "\n".join(
            [f"{feature}" for feature,definition in self.styles.items()]
        )
        
        prompt = (
            "Листа на различни стилски и лингвистички особини:\n\n"
            f"{features_text}\n\n"
            "Проучи го следниот пасус и за секоја од овие особини одговори со 'Да' или 'Не' дали авторот ја користи.\n\n"
            f"Пасус:\n{song}\n\n"
            "Одговор (форматирај по ред: 'Фигуративен јазик: Да', 'Сарказам: Не', ...):"
        )
        
        return prompt   
    def create_full_extraction_prompt_with_definition(self,song: str) -> str:
       
        features_text = "\n".join(
            [f"{feature}" for feature,_ in self.styles.items()]
        )
        prompt = (
            "Листа на различни стилски и лингвистички особини:\n\n"
            f"{features_text}\n\n"
            "Проучи го следниот пасус и за секоја од овие особини одговори со 'Да' или 'Не' дали авторот ја користи.\n\n"
            f"Пасус:\n{song}\n\n"
            "Одговор (форматирај по ред: 'Фигуративен јазик: Да', 'Сарказам: Не', ...):"
        )
        
        return prompt      
    def create_csv_for_extraction(self,
                                number_of_songs=10,
                                styles_path='api_styles_all_in_one_text.csv',
                                results_path='author_songs_to_create_only_with_styles.csv'):
        
        extracted_styles = pd.read_csv(styles_path)
        unique_authors = extracted_styles['author'].unique()

        columns = [
            'author',
            'name_of_sample_song',
            'styles',
            'name_of_new_song',
            'song',
            'input',
            'output',
            'total',
            'ms'
        ]

        if not os.path.exists(results_path):
            result_df = pd.DataFrame(columns=columns)
            result_df.to_csv(results_path, index=False)

        existing_df = pd.read_csv(results_path)

        new_rows = []
        for author in unique_authors:
            songs_for_author = self.extract_n_random_songs_with_styles_for_author(author, number_of_songs)
            
            for _, song_info in songs_for_author.iterrows():
                row = {
                    'author': song_info['author'],
                    'name_of_sample_song': song_info['song_title'],
                    'styles': song_info['extracted_styles'],
                    'name_of_new_song': '',
                    'song': '',
                    'input': '',
                    'output': '',
                    'total': '',
                    'ms': ''
                }
                new_rows.append(row)

        updated_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)

        updated_df.to_csv(results_path, index=False)
        print(f"✅ CSV updated/created at: {results_path}")  
    def create_prompt_template(self, author, all_author_words, example_song='', num_words=10, styles=None,dictionary=[],semanticly_similar_song=''):
        styles = [s.strip() for s in (styles or []) if isinstance(s, str) and s.strip()]

        
        most_common_words = []
        if all_author_words and 'expressive_words' in all_author_words and author in all_author_words['expressive_words']:
            most_common_words = all_author_words['expressive_words'][author]

        top_words = [word for word, _ in most_common_words[:num_words]]
        words_string = ", ".join(top_words)
        styles_string = "\n".join(f"- {s}" for s in styles)
        dictionary_string=''.join(dictionary)
        prompt_parts = []

        
        if styles:
            prompt_parts.append("Насоки на значење што треба да се искористат:")
            prompt_parts.append(styles_string)
        if most_common_words:
            prompt_parts.append("Најизразити зборови кои треба да се искористат во песната:")
            prompt_parts.append(words_string)
        if dictionary:
            prompt_parts.append('Најизрасити зборови кои треба да се искористат во песната и нивна дефиницја во толковен речник.')
            prompt_parts.append(dictionary_string)
        if not(styles) and  not(most_common_words)  and not(dictionary):
            prompt_parts.append(
            f"\n\nИзгенерирај македонска поезија, во стилот на {author}."
            "Песната мора да има наслов. Насловот запиши го во следниот формат: "
            "<НАСЛОВ>Тука вметни го насловот</НАСЛОВ>. "
            "Песната генерирај ја во рамките на <ПЕСНА>Тука вметни ја песната</ПЕСНА>. "
        )
            prompt = "\n".join(prompt_parts)
        prompt_parts.append(
            "\n\nИзгенерирај македонска поезија користејќи ги горенаведените насоки. "
            "Песната мора да има наслов. Насловот запиши го во следниот формат: "
            "<НАСЛОВ>Тука вметни го насловот</НАСЛОВ>. "
            "Песната генерирај ја во рамките на <ПЕСНА>Тука вметни ја песната</ПЕСНА>. "
            "Не ги користи имињата на самите насоки на значење. Биди креативен!"
        )
    
        
        if example_song:
            prompt_parts.append(f"\n\nПример песна од авторот {author}:")
            prompt_parts.append(str(example_song).strip())
        if semanticly_similar_song:
            prompt_parts.append(f"\n\nПример исечоци од песни што се семантички слични со авторот {author}:")
            prompt_parts.append(str(semanticly_similar_song).strip())

        prompt = "\n".join(prompt_parts)
        return prompt, "\n".join(styles) 
    def fill_csv(self, styles_from='all_styles_to_create.csv', model='claude', mode=1):
        system = 'Ти си Македонски разговорник наменет за генерирање на македонска поезија.'
        songs_to_apply = pd.read_csv(styles_from)
        
        mode_to_suffix = {
            1: 'idf_styles_examples.csv',
            2: 'idf_styles.csv',
            3: 'idf.csv',
            4: 'styles.csv',
            5: 'raw_author.csv',
            6:'explanatory_dictionary.csv',
            7:'idf_styles_rag_example.csv',
            8:'idf_styles_rag_example_makedonizer.csv',
            9:'idf_styles_explanatory_dictionary_makedonizer.csv'
        }
        suffix = mode_to_suffix.get(mode, 'output.csv')
        output_path = f'all_songs_{mode}_{model}_{suffix}'

        processed_songs = set()
        if os.path.exists(output_path):
            try:
                existing = pd.read_csv(output_path)
                if {'author', 'song_title'}.issubset(existing.columns):
                    processed_songs = set(
                        zip(existing['author'].astype(str), existing['song_title'].astype(str))
                    )
                    print(f"🔁 Found existing output file with {len(processed_songs)} processed songs. Resuming from last unprocessed one...")
                else:
                    print("⚠️ Output file found but missing expected columns. Starting from scratch.")
            except Exception as e:
                print(f"⚠️ Could not read existing output file: {e}")
        else:
            print("📄 No existing output file found. Starting fresh.")

        start_time = time.time()
        total_time = 0
        total_songs = len(songs_to_apply)
        all_author_words = self.analyze_author_text()

        for idx, row in songs_to_apply.iterrows():
            song_title = str(row['name_of_sample_song'])
            author = str(row['author'])

            if (author, song_title) in processed_songs:
                print(f"[{idx+1}/{total_songs}] ⏩ Skipping already processed '{song_title}' by '{author}'")
                continue

            print(f"[{idx+1}/{total_songs}] Processing '{song_title}' by '{author}'")

            extracted_styles = self.extract_style_pairs(row['styles'], only_present=True)
            styles_to_apply = list(extracted_styles.keys())

            if mode == 1:
                print(f'Mode {mode}: model {model} idf + styles + example 1200')
                example_song = self.extract_n_random_songs_for_author(row['author'], number_of_songs=1)
                example_song_text = "\n".join(map(str, example_song['song_text'].dropna()))
                prompt, styles_string = self.create_prompt_template(
                    author=author,
                    all_author_words=all_author_words,
                    styles=styles_to_apply,
                    example_song=example_song_text
                )
                print(prompt)
                return
            elif mode == 2:
                print(f'Mode {mode}: model {model} idf + styles 1200')
                prompt, styles_string = self.create_prompt_template(
                    author=author,
                    all_author_words=all_author_words,
                    styles=styles_to_apply,
                )
                print(prompt)
                return

            elif mode == 3:
                print(f'Mode {mode}: model {model} idf 1200')
                prompt, styles_string = self.create_prompt_template(
                    author=author,
                    all_author_words=all_author_words,
                    styles=[]
                )
                print(prompt)
                return
            elif mode == 4:
                print(f'Mode {mode}: model {model} styles 1200')
                
                prompt, styles_string = self.create_prompt_template(
                    author=author,
                    all_author_words=[],
                    styles=styles_to_apply
                )
                print(prompt)
                return
               
            elif mode == 5:
                print(f'Mode {mode}: model {model} testing author model knowledge 1200')
                prompt, styles_string = self.create_prompt_template(
                    author=author,
                    all_author_words=[],
                    styles=[]
                )
                print(prompt)
                return
            elif mode==6:
                print(f'Mode {mode}: model {model} lemas')
                words_for_author=all_author_words['expressive_words'][author]
                lemas=self.lemmas_from_text(words_for_author)
                lemas_explained=self.vector_db.query_lemmas(lemas)
                text_values = [entry.get("text", "") for entry in lemas_explained]
                
                csv_data = []
                for result in lemas_explained:
                    if result['source'] == 'semantic':
                        
                        csv_data.append({
                            'lemma': result['lemma'],
                            'lemma_found': result['lemma_found'],
                            'distance': result['distance']
                        })

                
                if csv_data:
                    self.write_to_semantic_search_csv(csv_data, csv_file="semantic_search_log.csv")
                

                prompt, styles_string = self.create_prompt_template(
                    author=author,
                    all_author_words=[],
                    styles=[],
                    dictionary=text_values
                    
                )
                print(prompt)
                return
            elif mode==7:
                print(f'Mode {mode}: model {model} idf + styles /+ example 1200')
                example_song = self.extract_n_random_songs_for_author(row['author'], number_of_songs=1)
                
                example_song_text = "\n".join(map(str, example_song['song_text'].dropna()))
                query=self.vector_db.query_database_semantic(example_song_text)
                
                prompt, styles_string = self.create_prompt_template(
                    author=author,
                    all_author_words=all_author_words,
                    styles=styles_to_apply,
                    semanticly_similar_song=query['documents'][0]
                )
                print(prompt)
                return
            elif mode==8:
                print(f'Mode {mode}: model {model} idf + styles /+ example 1200')
                example_song = self.extract_n_random_songs_for_author(row['author'], number_of_songs=1)
                
                example_song_text = "\n".join(map(str, example_song['song_text'].dropna()))
                query=self.vector_db.query_database_semantic(example_song_text,collection_name='makedonizer_poetry_db')
                
                prompt, styles_string = self.create_prompt_template(
                    author=author,
                    all_author_words=all_author_words,
                    styles=styles_to_apply,
                    semanticly_similar_song=query['documents'][0]
                )
                print(prompt)
                return
                
            elif mode==9:
                print(f'Mode {mode}: model {model} using lemmas makedonizer')
                words_for_author=all_author_words['expressive_words'][author]
                lemas=self.lemmas_from_text(words_for_author)
                lemas_explained=self.vector_db.query_lemmas(lemas,collection_name='macedonian_dictionary_macedonizer')
                text_values = [entry.get("text", "") for entry in lemas_explained]
                
                csv_data = []
                for result in lemas_explained:
                    if result['source'] == 'semantic_macedonizer':
                        
                        csv_data.append({
                            'lemma': result['lemma'],
                            'lemma_found': result['lemma_found'],
                            'distance': result['distance']
                        })

                
                if csv_data:
                    self.write_to_semantic_search_csv(csv_data, csv_file="semantic_search_makedonizer_log_2.csv")
                

                prompt, styles_string = self.create_prompt_template(
                    author=author,
                    all_author_words=[],
                    styles=[],
                    dictionary=text_values
                    
                )
                print(prompt)
                return
                
                
                
            success = False
            retries = 0
            max_retries = 3

            while not success and retries < max_retries:
                try:
                    start = time.time()

                    if model == 'nova':
                        result = self.invoke_nova_micro(prompt, system)
                    elif model == 'claude':
                        result = self.invoke_claude_model(prompt, system)
                    if not result or 'output' not in result or 'message' not in result['output']:
                        raise ValueError("Invalid API response")

                    self.write_to_csv(
                        author, song_title, styles_string, result,
                        output_path=output_path
                    )

                    elapsed = time.time() - start
                    total_time += elapsed
                    print(f"[{idx+1}/{total_songs}] ✅ Processed '{song_title}' - {elapsed:.2f}s (Total {total_time:.2f}s)")

                    wait_time = random.uniform(5, 10)
                    print(f"Waiting {wait_time:.2f}s before next song...")
                    time.sleep(wait_time)

                    success = True

                except Exception as e:
                    retries += 1
                    print(f"[{idx+1}/{total_songs}] ⚠️ Error processing '{song_title}': {e}")
                    traceback.print_exc()

                    if "ThrottlingException" in str(e):
                        wait_time = random.uniform(20, 40)
                        print(f"Throttled! Waiting {wait_time:.2f}s before retrying...")
                    elif "No song found. Try again!" in str(e):
                        print(f"No song found! Retrying...")
                    else:
                        wait_time = random.uniform(10, 20)
                        print(f"Retrying after {wait_time:.2f}s...")

                    time.sleep(wait_time)

            if not success:
                print(f"[{idx+1}/{total_songs}] ❌ Skipping '{song_title}' after {max_retries} failed attempts.")

        total_elapsed = time.time() - start_time
        print("\n🏁 All songs processed!")
        print(f"✅ Total songs in list: {total_songs}")
        print(f"✅ Already processed (skipped): {len(processed_songs)}")
        print(f"✅ Newly processed this run: {total_songs - len(processed_songs)}")
        print(f"🕒 Total runtime: {total_elapsed/60:.2f} minutes\n")

    def write_to_csv(self, author, song_title, styles_to_apply, result, output_path='author_songs_created_only_with_styles.csv'):
        text = result['output']['message']['content'][0]['text']

        input_tokens = result['usage']['inputTokens']
        output_tokens = result['usage']['outputTokens']
        total_tokens = result['usage']['totalTokens']

        ms = result['metrics']['latencyMs']

        
        title_match = re.search(r'<НАСЛОВ>\s*(.*?)\s*</НАСЛОВ>', text, re.DOTALL)
        name_of_new_song = title_match.group(1).strip() if title_match else 'no_title_found'

        
        song_match = re.search(r'<ПЕСНА>\s*(.*?)\s*</ПЕСНА>', text, re.DOTALL)
        
        song_content = song_match.group(1).strip() if song_match else 'no_song_found'
        if song_content=='no_song_found':
            raise ValueError("No song found for the specified author.")
        text = f"{name_of_new_song}\n\n{song_content.strip()}"

        row = {
            'author': author,
            'song_title': song_title,
            'styles_to_apply': styles_to_apply,
            'name_of_new_song': name_of_new_song,
            'new_song': text,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'ms': ms
        }

        api_csv = pd.DataFrame([row])
        file_exists = os.path.isfile(output_path)
        api_csv.to_csv(output_path, mode="a", index=False, header=not file_exists, encoding="utf-8")
    def extract_style_pairs(self, text, only_present=False):
        pattern = r"([^:]+):\s*(.+)"
        pairs = re.findall(pattern, text)

        result = {}
        for key, value in pairs:
            key_clean = key.strip()
            value_clean = value.strip().lower()

            present_value = "Да" if value_clean == "да" else "Не"

            if only_present and present_value != "Да":
                continue

            result[key_clean] = present_value

        return result
    def analyze_author_text(
    self,
    min_df=3,
    max_df=0.8904674508605334,
    max_features=4619,
    n_top_words=10,
    ngram_range=(1, 1),
    stop_words=None,
    skip_stopwords=True
):
       

        df = self.df.copy()
        df['song_text_processed'] = df['song_text'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

        author_corpus = df.groupby("author")["song_text_processed"].apply(lambda x: " ".join(x)).reset_index()

        results = {'common_words': {}, 'expressive_words': {}}

        
        default_stop_words = [
            'ќе', 'би', 'ко', 'го', 'како', 'ги', 'ми', 'ти', 'те', 'му', 'само',
            'зашто', 'таа', 'тие', 'нè', 'но', 'сè', 'со', 'по', 'ли', 'ој', 'ни',
            'ниту', 'pinterest', 'до', 'таа', 'ние', 'вие', 'тие', 'си','то','сме',
            'бил','јас','нека','кога','колку','тоа','дека','или','зар','ил','ме','со',
            'кој','кон','та','оваа','овој','тој','кај','се','туку','ние','вие','тие','нѐ'
        ]
        if stop_words is None:
            stop_words = default_stop_words

        for author in author_corpus['author']:
            texts = df[df['author'] == author]['song_text_processed']
            all_words = ' '.join(texts).split()

            
            author_names = author.lower().split()
            author_first_name = author_names[0] if len(author_names) > 0 else ''
            author_last_name = author_names[-1] if len(author_names) > 1 else ''

            if skip_stopwords:
                all_words = [word for word in all_words if word not in stop_words]

            all_words = [word for word in all_words if word not in [author_first_name, author_last_name]]

            word_counts = Counter(all_words)
            common_words = [(word, count) for word, count in word_counts.most_common(n_top_words)]
            results['common_words'][author] = common_words

        
        vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words if skip_stopwords else None,
            ngram_range=ngram_range,
            max_features=max_features
        )

        X = vectorizer.fit_transform(author_corpus["song_text_processed"])
        feature_names = vectorizer.get_feature_names_out()

        for idx, author in enumerate(author_corpus["author"]):
            author_names = author.lower().split()
            author_first_name = author_names[0] if len(author_names) > 0 else ''
            author_last_name = author_names[-1] if len(author_names) > 1 else ''

            tfidf_vector = X[idx].toarray()[0]
            top_indices = tfidf_vector.argsort()[-n_top_words * 2:][::-1]
            top_terms = [
                (feature_names[i], tfidf_vector[i])
                for i in top_indices
                if feature_names[i] not in [author_first_name, author_last_name]
            ]
            top_terms = top_terms[:n_top_words]
            results['expressive_words'][author] = top_terms

        print("\nMost Expressive Words per Author (TF-IDF Scores):")
        for author, words in results['expressive_words'].items():
            print(f"\n{author}:")
            for word, score in words:
                print(f"  {word}: {score:.3f}")

        return results
    def print_random_prompt(self):

        try:
            songs_to_apply = pd.read_csv('author_songs_to_create_only_with_styles.csv')

            random_row = songs_to_apply.sample(1).iloc[0]
            author = random_row['author']

            all_author_words = self.analyze_author_text()

            extracted_styles = self.extract_style_pairs(random_row['styles'], only_present=True)
            available_styles = list(extracted_styles.keys())

            if not available_styles:
                print(f"No styles found for author '{author}', skipping.")
                return

            
            
            prompt, styles_string = self.create_prompt_template(
                author=author,
                all_author_words=all_author_words,
                styles=available_styles
            )

            print("=== 🎭 Random Prompt Generated ===")
            print(f"Author: {author}")
            print("\n--- Prompt ---")
            print(prompt)
            print("-----------------------------")

            return prompt 

        except Exception as e:
            print(f"Error generating random prompt: {e}")
    def write_to_semantic_search_csv(self,data, csv_file="semantic_search_log.csv"):
        csv_exists = os.path.exists(csv_file)
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(['lemma', 'lemma_found', 'distance'])
            for entry in data:
                writer.writerow([
                    entry.get('lemma', ''),
                    entry.get('lemma_found', ''),
                    entry.get('distance', 0.0)
            ])
    def invoke_claude_model(self, prompt, system):
        response = self.client.converse(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            system=[{"text": system}],
            inferenceConfig={
                "maxTokens": 2000,
                "temperature": 1,
                "topP": 0.999
            }
        )
        return response
    def create_csv_with_perplexity(self,input_csv,column):
        
        df_input = pd.read_csv(input_csv)
        
        output_csv = f"{input_csv.rsplit('.', 1)[0]}_with_perplexity.csv"
        if os.path.exists(output_csv):
            print(f"Found existing {output_csv}. Resuming from existing data.")
            df_output = pd.read_csv(output_csv)
            if len(df_output) != len(df_input):
                print(f"Warning: {output_csv} has {len(df_output)} rows, expected {len(df_input)}. Reverting to input copy.")
                df_output = df_input.copy()
                df_output['perplexity'] = float('nan')
        else:
            df_output = df_input.copy()
            df_output['perplexity'] = float('nan')
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "ai-forever/mGPT"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,      
            low_cpu_mem_usage=True,         
            device_map="auto"               
        )
        model.eval()
        
        
        
        if not os.path.exists(output_csv):
            df_output.to_csv(output_csv, index=False)
        
        
        total_rows = len(df_output)
        
        
        for idx, row in tqdm(df_output.iterrows(), total=total_rows, desc="Processing rows"):
            
            if pd.notna(df_output.at[idx, 'perplexity']):
                print(f"Row {idx+1}/{total_rows}: Already processed (perplexity: {df_output.at[idx, 'perplexity']}).")
                continue
            
            text = row[column]
            if pd.notna(text) and isinstance(text, str) and text.strip():
                try:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs, labels=inputs["input_ids"])
                        loss = outputs.loss
                    perplexity = torch.exp(loss).item()
                    df_output.at[idx, 'perplexity'] = perplexity
                    print(f"Row {idx+1}/{total_rows}: Perplexity = {perplexity}")
                    
                    
                    if idx == 0:
                        df_output.iloc[[idx]].to_csv(output_csv, mode='w', index=False)  
                    else:
                        df_output.iloc[[idx]].to_csv(output_csv, mode='a', header=False, index=False)  
                except Exception as e:
                    print(f"Row {idx+1}/{total_rows}: Error calculating perplexity: {e}")
                    df_output.at[idx, 'perplexity'] = float('nan')
                    df_output.iloc[[idx]].to_csv(output_csv, mode='a', header=False, index=False)
            else:
                print(f"Row {idx+1}/{total_rows}: Skipping invalid or empty text")
                df_output.at[idx, 'perplexity'] = float('nan')
                df_output.iloc[[idx]].to_csv(output_csv, mode='a', header=False, index=False)
        
        print(f"Output saved to {output_csv}")
        if 'perplexity' not in df_input.columns:
            print("Original DataFrame unchanged (immutable).")
        else:
            print("Warning: Original DataFrame was modified unexpectedly.")
        
        return output_csv
    def lemmas_from_text(self, words):
        lemmas = []
        for word in words:
            doc = self.spacy(word[0])  
            for token in doc:
                if token.is_alpha:  
                    lemmas.append(token.lemma_.lower())
        return lemmas
    
        
    def plot_perplexity_histogram_n_bins(self,bins=13):
        folder_path = "final_results_csv"
        
        csv_files = glob.glob(os.path.join(folder_path, "*perplexity*.csv"))

        if not csv_files:
            print("No CSV files with 'perplexity' found.")
            return
        
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(csv_files)))  
        
        plt.figure(figsize=(10, 6))
        
        for file, color in zip(csv_files, colors):
            df = pd.read_csv(file)
            
            perplexity_col = next((col for col in df.columns if 'perplexity' in col.lower()), None)
            
            if perplexity_col is None:
                print(f"Skipping {file} — no 'perplexity' column found.")
                continue
            
            
            plt.hist(df[perplexity_col].dropna(), bins=bins, edgecolor='black', alpha=0.5, 
                    label=os.path.basename(file), color=color)
        
        if not plt.gca().has_data():
            print("No valid perplexity data found.")
            return
        
        plt.title("Perplexity Distribution Across CSV Files (13 Bins)")
        plt.xlabel("Perplexity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    def do_not_use_plot_perplexity_histogram_auto_bins(self):
        folder_path = "final_results_csv"
        
        csv_files = glob.glob(os.path.join(folder_path, "*perplexity*.csv"))

        if not csv_files:
            print("No CSV files with 'perplexity' found.")
            return
        
        
        all_perplexities = []
        for file in csv_files:
            df = pd.read_csv(file)
            
            perplexity_col = next((col for col in df.columns if 'perplexity' in col.lower()), None)
            
            if perplexity_col is None:
                print(f"Skipping {file} — no 'perplexity' column found.")
                continue
            
            all_perplexities.extend(df[perplexity_col].dropna().values)
        
        if not all_perplexities:
            print("No valid perplexity data found.")
            return
        
        
        data = np.array(all_perplexities)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)  
        n = len(data)
        bin_width = 2 * iqr / (n ** (1/3))  
        if bin_width == 0: 
            print("Data has no variability; using default bins.")
            num_bins = 10
        else:
            data_range = np.max(data) - np.min(data)
            num_bins = int(np.ceil(data_range / bin_width))  
            num_bins = max(5, min(num_bins, 100)) 
        
        
        plt.figure(figsize=(10, 6))
        plt.hist(all_perplexities, bins=num_bins, edgecolor='black', alpha=0.7, label='Perplexity Values')
        
        plt.title(f"Histogram of Perplexity Values (Auto Bins: {num_bins})")
        plt.xlabel("Perplexity Range")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    def plot_perplexity_histogram_fixed_bins(self):
        folder_path = "final_results_csv"
        
        csv_files = glob.glob(os.path.join(folder_path, "*perplexity*.csv"))

        if not csv_files:
            print("No CSV files with 'perplexity' found.")
            return
        
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(csv_files))) 
        
        
        bin_edges = np.arange(0, 131, 5)  
        
        plt.figure(figsize=(10, 6))
        
        for file, color in zip(csv_files, colors):
            df = pd.read_csv(file)
            
            perplexity_col = next((col for col in df.columns if 'perplexity' in col.lower()), None)
            
            if perplexity_col is None:
                print(f"Skipping {file} — no 'perplexity' column found.")
                continue
            
            
            plt.hist(df[perplexity_col].dropna(), bins=bin_edges, edgecolor='black', alpha=0.5, 
                    label=os.path.basename(file), color=color)
        
        if not plt.gca().has_data():
            print("No valid perplexity data found.")
            return
        
        plt.title("Perplexity Distribution Across CSV Files (0-5, 5-10, ..., 125-130)")
        plt.xlabel("Perplexity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    def plot_perplexity_histogram_with_kde(self):
        folder_path = "final_results_csv"
        
        csv_files = glob.glob(os.path.join(folder_path, "*perplexity*.csv"))

        if not csv_files:
            print("No CSV files with 'perplexity' found.")
            return
        
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(csv_files)))  
        
        
        bin_edges = np.arange(0, 131, 5)  
        
        plt.figure(figsize=(10, 6))
        
        for file, color in zip(csv_files, colors):
            df = pd.read_csv(file)
            
            perplexity_col = next((col for col in df.columns if 'perplexity' in col.lower()), None)
            
            if perplexity_col is None:
                print(f"Skipping {file} — no 'perplexity' column found.")
                continue
            
            
            data = df[perplexity_col].dropna()
            if len(data) == 0:
                continue
            
            
            plt.hist(data, bins=bin_edges, edgecolor='black', alpha=0.3, 
                    label=f"{os.path.basename(file)} (hist)", color=color, density=True)
            
            
            sns.kdeplot(data=data, color=color, label=f"{os.path.basename(file)} (KDE)", linewidth=2)
        
        if not plt.gca().has_data():
            print("No valid perplexity data found.")
            return
        
        plt.title("Perplexity Distribution with KDE Across CSV Files (0-5, 5-10, ..., 125-130)")
        plt.xlabel("Perplexity")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    def plot_perplexity_kde_only(self):
        folder_path = "final_results_csv"
        
        csv_files = glob.glob(os.path.join(folder_path, "*perplexity*.csv"))

        if not csv_files:
            print("No CSV files with 'perplexity' found.")
            return
        
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(csv_files)))  
        
        plt.figure(figsize=(10, 6))
        
        for file, color in zip(csv_files, colors):
            df = pd.read_csv(file)
            
            perplexity_col = next((col for col in df.columns if 'perplexity' in col.lower()), None)
            
            if perplexity_col is None:
                print(f"Skipping {file} — no 'perplexity' column found.")
                continue
            
            
            data = df[perplexity_col].dropna()
            if len(data) == 0:
                continue
            
            
            sns.kdeplot(data=data, color=color, label=os.path.basename(file), linewidth=2, clip=(0, 130))
        
        if not plt.gca().has_data():
            print("No valid perplexity data found.")
            return
        plt.title("Perplexity KDE Across CSV Files (0 to 130)")
        plt.xlabel("Perplexity")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
st=StyleTransfer()
from datetime import datetime
now = datetime.now()
print("Current date and time:", now)
#st.plot_perplexity_kde_only()
st.create_csv_with_perplexity('/home/ivan/Desktop/Diplomska/all_songs_9_nova_idf_styles_explanatory_dictionary_makedonizer.csv','new_song')
st.fill_csv(model='nova_1',mode=7)
#now = datetime.now()
print("Current date and time:", now)    
#3 1:24 - 6.22
#4 7:06-10:50 16:25 -18 : 54
#5 20:22 11:45 
#6 -6:53 duration (308)
#7 11:53 - 15:50    pred- 17:55 (77 minuti)
#8 19:36 -00 36 33 
#9 20:20-1:35

# nova micro 
#5 14:39     19:59 - 5:20 sati celosno (imashe lufta)
#4 20:51  - 00:39 -4:50 sati celosno
#3 0:40 - 4:01 -194 
#2 5:56:20      - 9:35
#1 10:10   14:10
#6 15:47 22:15:54
#7 19:42 02:29: 
#8 8:10 - 15:35
#9 3:20 - 10:13
