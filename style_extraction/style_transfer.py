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

        with open(log_path, "a", encoding="utf-8") as log_file:
            for i, (_, row) in enumerate(sf_styles.iterrows()):
                target_feature = row['style_feature_category']
                target_feature_definition = self.styles.get(target_feature, "")
                
                example_1="Особина:Сарказам\nОригинално: Денес работев цел ден без пауза.\nСо „Сарказам“: О, прекрасно, токму тоа ми требаше — уште еден ден без одмор!\n"
                example_2="Особина:Активен глас\nОригинално: Писмото беше испратено од мене.\nСо „Активен глас“: Јас го испратив писмото\n"
                
                user_message = (
                    f"{target_feature_definition}\n Ова ја претставува дефиницијата, не ја давај нејзе, во твојот одговор\n\n"
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
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "repetition_penalty": 2,
                    'early_stopping':True,
                    "frequency_penalty": 0.2,
                    "presence_penalty": 0.15,
                    'stop': ["<|im_end|>"],
                    "max_tokens":len(song),
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
        print(len(sf_selected))
        print(len(st_selected))
        return self.apply_styles_iterative(sf_styles=sf_selected,st_song_text=st_song_text,st_styles=st_selected)
    def extract_style_from_all_songs(self, songs_csv, output_filename="extracted_styles_1.csv", save_every=20):    
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
            
st = StyleTransferLocal(model="http://127.0.0.1:8080/v1/chat/completions")
#st.extract_style_from_all_songs("classification/cleaned_songs.csv",'vezilka_test.cvs')
molitva_teskts="""Молитва – Гане Тодоровски

(пред крајот на годината
и пред истекот на векот)

Боже, зарем ќе оставиш да бидам неразбран
Од современиците мои – што ги мунѕосував ко џган!
Зарем ќе оставиш да останам во уплав збран
И да си заминам од веков – од мунѕосаните мунѕосан?
Придај им на моите сотатковинци додатен ум,
За да ме доразберат, и да ме следат молчешкум;
Не ги прекорувај престрого, не кревај ненужен шум,
Поучи ги, кога зборувам, да стојат отпростум!

За да се знае, конечно еднаш, КОЈ е КОЈ?
За да не понесам вина, дека, дури бев жив
Малцина надзборев а трижтолкумина не победив!

Господе, дај искористи го авторитетот свој,
Па, додека е време, застани на моја страна,
За да поверувам дека ѝ бев на вистината бранач!

Москва, декември 1994 г."""
st_song_title='Молитва'
print(st.transfer_style('Петре М. Андреевски','Наопачно оро','Гане Тодоровски',molitva_teskts,st_song_title))
