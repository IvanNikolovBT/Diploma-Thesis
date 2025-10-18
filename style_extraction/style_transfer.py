import sys
import os
import pandas as pd
import requests
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from poetry_DB import PoetryDB
import time
from tqdm import tqdm
import datetime
import boto3
import json
import re
class StyleTransferLocal:
    
    def __init__(self,model="trajkovnikola/MKLLM-7B-Instruct"):

        self.system={"role": "system","content": 
            ("[INST]Разговор помеѓу корисник и разговорник за екстракција на стил македонска поезија. Асистентот дава корисни, детални и љубезни одговори на прашањата на корисникот.Ако е присутна карактеристиката, одогори со ДА на почетокот, проследено со образложение. Ако не е присутна, одговори само со НЕ и ништо друго.[/INST]</s>.")}        
        self.db=PoetryDB()
        self.CSV_PATH="classification/cleaned_songs.csv"
        self.df=pd.read_csv(self.CSV_PATH)
        self.random_seed=47
        self.styles_path='style_extraction/vezilka_test.cvs'
        self.model=model
        self.styles=self.load_styles()
        self.client = boto3.client("bedrock-runtime",region_name="eu-central-1",)
    
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
    def extract_n_random_styles_for_author(self, author_name='Блаже Конески', number_of_songs=10):
        styles=pd.read_csv('api_styles_all_in_one_text.csv')
        author_songs = styles[styles['author'] == author_name]
        return author_songs.sample(
            n=min(number_of_songs, len(author_songs)),
            random_state=None  
        )
    def extract_all_songs_for_author(self, author_name='Блаже Конески'):
        return self.df[self.df['author'] == author_name]
    
    def extract_style_from_song(self,song,target_feature,target_feature_definition):
        
        user_message_mk_llm = (
            f"{target_feature_definition}\n"
            "Ова ја претставува дефиницијата, не ја давај нејзе, во твојот одговор.\n\n"
            f"Напиши краток опис дали авторот на следниот пасус ја содржи оваа особина: {target_feature}.\n"
            "Одговори со Да или Не.\n\n"
            "Следуваат неколку примери:\n\n"

            "Пример 1:\n"
            "Особина: Сарказам\n"
            "Пасус: „О, прекрасно! Баш сакав да ми се расипе телефонот сред бел ден.“\n"
            "Опис: Авторот зборува иронично, изразувајќи спротивно од тоа што мисли. Одговор: Да.\n\n"

            "Пример 2:\n"
            "Особина: Сарказам\n"
            "Пасус: „Мојот телефон се расипа денес и тоа ми го уништи денот.“\n"
            "Опис: Авторот директно го изразува своето незадоволство без иронија или потсмев. Одговор: Не.\n\n"

            
            "Пример 3:\n"
            "Особина: Активен глас\n"
            "Пасус: „Јас ја напишав песната за еден час.“\n"
            "Опис: Авторот користи активен глас каде што подметот ја извршува дејството. Одговор: Да.\n\n"

            f"Пасус:\n{song}\n\nОпис:"
        )
        user_message_vezilka = (
            f"{target_feature_definition}\n"
            "Ова ја претставува дефиницијата, не ја давај нејзе, во твојот одговор.\n\n"
            f"Одговори на пашањето дали авторот на следниот пасус ја содржи оваа особина: {target_feature}.\n"
            "Одговори со Да,присутно! или Не!.\n\n"
            "Биди многу строг при својата одлука, ако си сигурен дека е присутна карактеристиката, тогаш запиши одогори со да."
            f"Пасус:\n{song}\n\nОпис:"
        )
        messages = [self.system,{"role": "user", "content": user_message_vezilka}]

        payload_mk_llm = {
        "model": self.model,
        "messages": messages,
        "temperature": 0.3,
        "repetition_penalty":2,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.3,
        "top_p": 0.9,
        "max_tokens":200,
        "stop": ["\n","\n\n","<|im_end|>"]}
        payload_vezilka = {
        "model": self.model,
        "messages": messages,
        "temperature": 0.4,
        "top_p": 0.9,
        "max_tokens":200,
        "stop": ["\n","\n\n","<|im_end|>"]}
        resp = requests.post("http://127.0.0.1:8080/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload_vezilka)
        
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
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
        
    def write_to_csv(self,author:str,song_title:str,result:json,output_path='api_styles_all_in_one_text.csv'):
        
        text=result['output']['message']['content'][0]['text'] 
        
        stop_reason=result['stopReason']           
        
        input_tokens=result['usage']['inputTokens']
        output_tokens=result['usage']['outputTokens']
        total_tokens=result['usage']['totalTokens']
        
        ms=result['metrics']['latencyMs']
        
        row={'author':author,
             'song_title':song_title,
             'extracted_styles':text,
             'input_tokens':input_tokens,
             'output_tokens':output_tokens,
             'total_tokens':total_tokens,
             'ms':ms
             }
        api_csv = pd.DataFrame([row])
        file_exists = os.path.isfile(output_path)
        api_csv.to_csv(output_path, mode="a", index=False, header=not file_exists, encoding="utf-8")
        
    def extract_styles_from_song_using_api(self,song,author,song_title,output_path,without_def=True):
        system="Ти си разговорник за екстракција на стил на македонска поезија."
        if without_def:
            prompt=self.create_full_prompt_without_definition(song)
        else:
            prompt=self.create_full_prompt_with_definition(song)
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
        
    def extract_style_from_songs(self, sample_songs=[], output_dir=''):
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, "extracted_styles.csv") if output_dir else "extracted_styles.csv"

        total_time=0
        file_exists = os.path.exists(output_path)

        for _, row in sample_songs.iterrows():
            author = row["author"]
            song = row["song_title"]
            original_song = row["song_text"]

            for category, definition in self.styles.items():
                start_time = time.time()

                extracted_text=self.extract_style_from_song(song,category,definition)
                
                end_time = time.time()
                time_needed = end_time - start_time

                
                df_row = pd.DataFrame([{
                    "author": author,
                    "song": song,
                    "style_feature_category": category,
                    "extracted_text": extracted_text,
                    #"original_song": original_song,
                    "time_needed": time_needed
                }])
                total_time+=time_needed
                
                df_row.to_csv(output_path, mode="a", index=False, header=not file_exists)
                file_exists = True

                
                print(f"[SAVED] Author='{author}', Song='{song}', Style='{category}', Time={time_needed:.2f}s")
            
            print(f'Total time {total_time:.2f}')
    def iterate_over_author(self,author):
        songs=self.extract_n_random_songs_for_author(author_name=author,number_of_songs=1)
        self.extract_style_from_songs(songs)

    def get_present_styles_for_song(self, title, author):
        self.df = pd.read_csv(self.styles_path)
        pattern = r'^Да' 
        return self.df[
            (self.df['author']== author) & (self.df['song']==title) &
            (self.df['extracted_text'].str.match(pattern, na=False))
        ]
    def apply_styles_iterative(self, sf_styles, st_song_text, st_song_title, st_author,st_styles, log_path=None):
        if log_path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"song_style_log_{st_song_title or 'song'}_{ts}.txt"
            log_path = "".join(c for c in log_path if c.isalnum() or c in (' ','.','_','-')).rstrip()

        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        new_song=st_song_text
        list1=sf_styles['style_feature_category'].unique().tolist()
        list2=st_styles['style_feature_category'].unique().tolist()
        diff = [item for item in list1 if item not in list2]
        print(diff)
        print(st_song_text)
        cumulative=0
        with open(log_path, "a", encoding="utf-8") as log_file:
            i=0
            for style in diff:
                target_dif=self.styles[style]
                user_message = (
                        f"{target_dif}\n Ова ја претставува дефиницијата, не ја давај нејзе, во твојот одговор\n\n"
                        f"Искористи ја оваа особина {style} ВРЗ песната: .\n"
                        f"Следувват два примери:\n"
                        f"Пасус:\n{new_song}\n\n Обработена песна: <Генерирај ја песната тука ве замолувам>"
                    )
                new_system={"role": "system","content": 
                    ("[INST]Разговор помеѓу корисник и разговорник  за применување на  стил македонска поезија. Асистентот дава песна со применет стил по барање на корисникот.[/INST]</s>.")}
                    
                messages = [new_system, {"role": "user", "content": user_message}]

                payload = {
                        "model": self.model,
                        "messages": messages,
                        "top_p": 0.9,
                        'early_stopping':True,
                        "frequency_penalty": 0.2,
                        "presence_penalty": 0.15,
                        'stop': ["<|im_end|>"],
                        "max_tokens":int(1.5*len(st_song_text)),
                    }
                start_time=time.time()
                resp = requests.post(
                            "http://127.0.0.1:8080/v1/chat/completions",
                            headers={"Content-Type": "application/json"},
                            json=payload,
                            timeout=200
                        )
                duration = time.time() - start_time
                cumulative += duration
                data = resp.json()
                new_song = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                print("\n" + "="*40)
                print(f"Iteration {i+1}/{len(diff)} - style added: {target_dif}")
                print(f"Duration: {duration:.3f} s")
                print("-" * 40)
                print(new_song)
                print("="*40 + "\n")
                i+=1
                log_file.write(f"--- Iteration {i+1} / {len(diff)} ---\n")
                log_file.write(f"Style added: {target_dif}\n")
                log_file.write(f"Author: {st_author}\n")
                log_file.write(f"Song title: {st_song_title}\n")
                log_file.write(f"Duration (s): {duration:.3f}\n")
                log_file.write(f"Cumulative duration (s): {cumulative:.3f}\n")
                log_file.write("Resulting song:\n")
                log_file.write(new_song + "\n\n")
                log_file.flush()
    def create_full_prompt_without_definition(self,song: str) -> str:

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
    def create_full_prompt_with_definition(self,song: str) -> str:
       
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
    def apply_styles_iterative_(self, sf_styles, st_song_text, st_song_title, st_author,st_styles, log_path=None):
    
        song = st_song_text

        
        if log_path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"song_style_log_{st_song_title or 'song'}_{ts}.txt"
            log_path = "".join(c for c in log_path if c.isalnum() or c in (' ','.','_','-')).rstrip()

        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        cumulative = 0.0
        total_styles = len(sf_styles)
        target_features = []
        target_feature_definitions = []
        for _, row in sf_styles.iterrows():
            target_feature = row['style_feature_category']
            if target_feature not in st_styles['style_feature_category'].values:
                target_features.append(target_feature)
                target_feature_definitions.append(self.styles[target_feature])
            
        
        if not target_features:
            return song
        i=0
        with open(log_path, "a", encoding="utf-8") as log_file:
            for target_feature,target_definition in zip(target_features,target_feature_definitions):
                
                
                example_1="Особина:Сарказам\nОригинално: Денес работев цел ден без пауза.\nСо „Сарказам“: О, прекрасно, токму тоа ми требаше — уште еден ден без одмор!\n"
                example_2="Особина:Активен глас\nОригинално: Писмото беше испратено од мене.\nСо „Активен глас“: Јас го испратив писмото\n"
                
                user_message = (
                    f"{target_definition}\n Ова ја претставува дефиницијата, не ја давај нејзе, во твојот одговор\n\n"
                    f"Искористи ја оваа особина {target_feature} ВРЗ песната: .\n"
                    f"Како одговор врати ја назад песната, но со применет {target_feature} врз нејзе. Оваа е клучно.\n"
                    f"Следувват два примери:\n"
                    f'{example_1}'
                    f'{example_2}'
                    f"Пасус:\n{song}\n\n Обработена песна:"
                )
                
               
                new_system={"role": "system","content": 
                ("[INST]Разговор помеѓу корисник и разговорник  за применување на  стил македонска поезија. Асистентот дава корисни, детални и љубезни одговори на прашањата на корисникот.[/INST]</s>.")}
                #probaj so indivdualni primeri pomali 
                #predefinirani stilovi na ekstrakjcija i da proveri dali e prisuten
                # (/) vsushnost razlika na stilovi
                #ostavi default
                #podobro odednash site i da se zpaishi akko zakluchok
                messages = [new_system, {"role": "user", "content": user_message}]

                payload = {
                    "model": self.model,
                    "messages": messages,
                    "top_p": 0.9,
                    'early_stopping':True,
                    "frequency_penalty": 0.2,
                    "presence_penalty": 0.15,
                    'stop': ["<|im_end|>"],
                    "max_tokens":int(1.5*len(song)),
                }
                """"output_ids = self.model.generate(input_ids=data_x_input_ids,
                                                 attention_mask=data_x_attention_mask,
                                                 max_new_tokens=max_length,
                                                 eos_token_id=tokenizer.eos_token_id,
                                                 pad_token_id=tokenizer.pad_token_id,
                                                 early_stopping=True,
                                                 num_return_sequences=1,
                                                 # no_repeat_ngram_size=2,
                                                 # repetition_penalty=2.0,
                                                 do_sample=False,
                                                 # top_p=0.5
                                                 )"""
                
                start_time = time.time()
                try:
                    resp = requests.post(
                        "http://127.0.0.1:8080/v1/chat/completions",
                        headers={"Content-Type": "application/json"},
                        json=payload,
                        timeout=200
                    )
                    duration = time.time() - start_time

                    if resp.ok:
                        data = resp.json()
                        new_song = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

                        if not new_song:
                            new_song = song
                            note = f" (empty response — kept previous song text)"
                        else:
                            note = ""
                    else:
                        new_song = song
                        note = f" (request failed: status {resp.status_code})"

                except Exception as e:
                    duration = time.time() - start_time
                    new_song = song
                    note = f" (exception during request: {e})"

                cumulative += duration
                song = new_song

                
                print("\n" + "="*40)
                print(f"Iteration {i+1}/{total_styles} - style added: {target_feature}")
                print(f"Duration: {duration:.3f} s{note}")
                print("-" * 40)
                print(song)
                print("="*40 + "\n")
                i+=1
                log_file.write(f"--- Iteration {i+1} / {total_styles} ---\n")
                log_file.write(f"Style added: {target_feature}\n")
                log_file.write(f"Author: {st_author}\n")
                log_file.write(f"Song title: {st_song_title}\n")
                log_file.write(f"Duration (s): {duration:.3f}\n")
                log_file.write(f"Cumulative duration (s): {cumulative:.3f}\n")
                if note:
                    log_file.write(f"Note: {note}\n")
                log_file.write("Resulting song:\n")
                log_file.write(song + "\n\n")
                log_file.flush()

        return song, log_path
    def apply_styles_all_at_once(self, sf_styles, st_song_text,st_styles):
        song = st_song_text
        max_words = len(st_song_text.split()) 
        target_features = []
        target_feature_definitions = []
        
        
        for _, row in sf_styles.iterrows():
            target_feature = row['style_feature_category']
            if target_feature not in st_styles['style_feature_category'].values:
                target_features.append(target_feature)
                target_feature_definitions.append(self.styles[target_feature])
        
        
        if not target_features:
            return song
        print(target_features)
        definitions_text = "Дефиниции на особини:\n" + "\n".join(
            [f"{i+1}. {target_features[i]} – {target_feature_definitions[i]}" for i in range(len(target_features))]
        )
        
        example_1 = (
            "Особина: Сарказам\n"
            "Оригинално: Денес работев цел ден без пауза.\n"
            "Со „Сарказам“: О, прекрасно, токму тоа ми требаше — уште еден ден без одмор!\n"
        )
        example_2 = (
            "Особина: Активен глас\n"
            "Оригинално: Писмото беше испратено од мене.\n"
            "Со „Активен глас“: Јас го испратив писмото.\n"
        )
        
       
        styles_list = ", ".join(target_features)
        user_message = (
            f"{definitions_text}\n"
            f"Овие се  дефинициите на стиловите, не ги давај нив во твојот одговор.\n\n"
            f"Искористи ги сите овие особини: {styles_list} ВРЗ песната истовремено.\n"
            f"Како одговор врати ја назад песната, но со применети {styles_list} врз нејзе. Оваа е клучно.\n"
            f"Следуваат два примери:\n"
            f'{example_1}\n'
            f'{example_2}\n'
            f"Пасус:\n{song}\n\nОбработена песна со применети стилови:"
        )
        
        new_system = {
            "role": "system",
            "content": "[INST]Разговор помеѓу корисник и разговорник за применување на стил македонска поезија. Асистентот дава корисни, детални и љубезни одговори на прашањата на корисникот.[/INST]</s>"
        }
        messages = [new_system, {"role": "user", "content": user_message}]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.5,
            "top_p": 0.9,
            "stop": ["\n\n\n","<|im_end|>"],
            "max_tokens":int(1.5*max_words),
  
        }
        
        start = time.time()
        resp = requests.post(
            "http://127.0.0.1:8080/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        print(f'Time needed for applying all styles: {time.time() - start}')
        
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    def transfer_style(self,sf_author,sf_song_title,st_author,st_song_text,st_song_title): 
        #self,sf_author,sf_song_title):
        sf_selected=self.get_present_styles_for_song(sf_song_title,sf_author)
        st_selected=self.get_present_styles_for_song(st_song_title,st_author)
        return self.apply_styles_iterative(sf_styles=sf_selected,st_song_text=st_song_text,st_styles=st_selected,st_song_title=st_song_title,st_author=st_author)
    def extract_style_from_all_songs(self, songs_csv, output_filename="extracted_styles_1.csv", save_every=100):    
        if not os.path.exists(songs_csv):
            raise FileNotFoundError(f"CSV file not found: {songs_csv}")
        sample_songs = pd.read_csv(songs_csv)
        print(f"📄 Loaded {len(sample_songs)} songs from {songs_csv}")

        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, output_filename)
        
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path)
            processed = set(zip(
                existing_df["author"],
                existing_df["song"],
                existing_df["style_feature_category"]
            ))
            print(f"🔁 Found existing file with {len(processed)} processed entries. Resuming...")
            file_exists = True
        else:
            processed = set()
            file_exists = False
            print("🆕 No existing file found. Starting fresh.")

        buffer = []
        total_time = 0.0
        counter = 0

        total_items = len(sample_songs) * len(self.styles)
        pbar = tqdm(total=total_items, desc="Extracting styles", ncols=100)

        for _, row in sample_songs.iterrows():
            author = row["author"]
            song = row["song_title"]
            

            for category, definition in self.styles.items():
                key = (author, song, category)
                if key in processed:
                    pbar.update(1)
                    continue

                start_time = time.time()
                try:
                    extracted_text = self.extract_style_from_song(song, category, definition)
                except Exception as e:
                    print(f"❌ Error processing {key}: {e}")
                    pbar.update(1)
                    continue

                time_needed = time.time() - start_time
                total_time += time_needed

                buffer.append({
                    "author": author,
                    "song": song,
                    "style_feature_category": category,
                    "extracted_text": extracted_text,
                    "time_needed": time_needed
                })
                processed.add(key)
                counter += 1
                pbar.update(1)

                
                if counter % save_every == 0:
                    pd.DataFrame(buffer).to_csv(output_path, mode="a", index=False, header=not file_exists)
                    file_exists = True
                    buffer = []
                    print(f"💾 Progress saved ({counter} items processed so far).")

        
        if buffer:
            pd.DataFrame(buffer).to_csv(output_path, mode="a", index=False, header=not file_exists)
            print("💾 Final save completed.")

        pbar.close()
        print(f"\n🏁 Extraction complete. Total time: {total_time:.2f}s")  
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
            songs_for_author = self.extract_n_random_styles_for_author(author, number_of_songs)
            
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
    
    def only_styles_prompt(self,*key_lists):
  
        all_keys = set()
        for keys in key_lists:
            all_keys.update(keys)

        sorted_keys = sorted(all_keys)

        prompt = "Стилски фигури што треба да се искористат:\n"
        prompt += "\n".join(f"- {key}" for key in sorted_keys)
        prompt+="\nИзгенерирај македонска поезија користејќи ги горе наведените стилски фигури.  Песната мора да има наслов. Насловот запиши го во следниот формат <НАСЛОВ>Тука вметни го насловот </НАСЛОВ>"
        return prompt
    def fill_csv_using_only_styles(self):
        system = 'Ти си Македонски разговорник наменет за генерирање на македонска поезија.'
        songs_to_apply = pd.read_csv('author_songs_to_create_only_with_styles.csv')
        
        start_time = time.time()
        
        total_songs = len(songs_to_apply)
        
        for idx, row in songs_to_apply.iterrows():
            try:
                extracted_styles = self.extract_style_pairs(row['styles'], only_present=True)
                styles_to_apply = extracted_styles.keys()
                prompt = self.only_styles_prompt(styles_to_apply)
                
                result = self.invoke_nova_micro(prompt=prompt, system=system)
                
               
                if not result or 'output' not in result or 'message' not in result['output']:
                    print(f"[{idx+1}/{total_songs}] Warning: No valid reply from API for song '{row['name_of_sample_song']}' by '{row['author']}'")
                    continue
                
                self.write_to_csv_only_styles(
                    row['author'],
                    row['name_of_sample_song'],
                    row['styles'],
                    result
                )
                
                elapsed = time.time() - start_time
                print(f"[{idx+1}/{total_songs}] Processed '{row['name_of_sample_song']}' by '{row['author']}' - Time elapsed: {elapsed:.2f}s")
            
            except Exception as e:
                print(f"[{idx+1}/{total_songs}] Error processing '{row['name_of_sample_song']}' by '{row['author']}': {e}")

    def write_to_csv_only_styles(self, author, song_title, styles_to_apply, result, output_path='author_songs_created_only_with_styles.csv'):
        text = result['output']['message']['content'][0]['text']

        input_tokens = result['usage']['inputTokens']
        output_tokens = result['usage']['outputTokens']
        total_tokens = result['usage']['totalTokens']

        ms = result['metrics']['latencyMs']

        # Extract title from <НАСЛОВ> tags
        match = re.search(r'<НАСЛОВ>\s*(.*?)\s*</НАСЛОВ>', text, re.DOTALL)
        if match:
            name_of_new_song = match.group(1).strip()
            # Remove the tags from the text
            text = re.sub(r'<НАСЛОВ>\s*.*?\s*</НАСЛОВ>', '', text, flags=re.DOTALL)
        else:
            name_of_new_song = 'no_title_found'

        # Prepend the title at the beginning of the text
        text = f"{name_of_new_song}\n\n{text.strip()}"

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

st = StyleTransferLocal(model="http://127.0.0.1:8080/v1/chat/completions")
st.fill_csv_using_only_styles()

