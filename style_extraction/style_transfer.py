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
class StyleTransfer:
    
    def __init__(self):
        self.system_prompt={"role": "system","content": ("<s>[INST]Разговор помеѓу љубопитен корисник и разговорник за екстракција на стил македонска поезија. Асистентот дава корисни, детални и љубезни одговори на прашањата на корисникот.[/INST]</s>.")}
        self.styles_map = {
        "Фигуративен јазик": "Фигуративен јазик – употреба на метафори, симболи или алегории за да се пренесе значење кое не е буквално.",
        "Сарказам": "Сарказам – изразување на спротивното од она што навистина се мисли, со цел иронија или потсмев.",
        "Фрагмент од реченица": "Фрагмент од реченица – недовршена реченица без сите неопходни граматички делови.",
        "Долга реченица (run-on)": "Долга реченица (run-on) – реченица составена од повеќе мисли без правилна интерпункција или сврзници.",
        "Активен глас": "Активен глас – конструкција во која подметот ја врши дејството (на пр. „Јас го пишувам писмото“).",
        "Пасивен глас": "Пасивен глас – конструкција во која подметот ја прима дејството (на пр. „Писмото е напишано од мене“).",
        "Грешки во согласување": "Грешки во согласување – несогласување помеѓу подмет и прирок во број, род или лице.",
        "Машки заменки": "Машки заменки – зборови што се користат за замена на машки именки (тој, него, негов).",
        "Женски заменки": "Женски заменки – зборови што се користат за замена на женски именки (таа, неа, нејзин).",
        "Просоцијално однесување": "Просоцијално однесување – јазик кој покажува соработка, помош или емпатија.",
        "Антисоцијално однесување": "Антисоцијално однесување – јазик кој покажува агресија, непочитување или отфрлање на норми.",
        "Бидување љубезен": "Бидување љубезен – употреба на изрази на почит, културни и учтиви форми.",
        "Прикажување меѓучовечки конфликт": "Прикажување меѓучовечки конфликт – јазик што одразува несогласување, расправија или тензија.",
        "Морализирање": "Морализирање – изразување на судови врз основа на морални вредности.",
        "Зборови за комуникација": "Зборови за комуникација – зборови поврзани со говор, пишување или размена на пораки.",
        "Индикатори за моќ": "Индикатори за моќ – зборови што покажуваат авторитет, доминација или контрола.",
        "Говор за постигнување": "Говор за постигнување – јазик кој изразува цел, амбиција или успех.",
        "Индикација на сигурност": "Индикација на сигурност – зборови што изразуваат апсолутна увереност.",
        "Бидување несигурен/колеблив": "Бидување несигурен/колеблив – зборови што изразуваат сомнеж, несигурност или условност.",
        "Увид": "Увид – јазик што покажува самосвест, разбирање или размислување.",
        "Сè или ништо размислување": "Сè или ништо размислување – изрази со апсолутни термини без нијанси.",
        "Зборови поврзани со меморија": "Зборови поврзани со меморија – зборови кои се однесуваат на сеќавање или минати искуства.",
        "Позитивна емоција": "Позитивна емоција – зборови што пренесуваат радост, љубов, ентузијазам.",
        "Негативна емоција": "Негативна емоција – зборови што пренесуваат тага, болка или огорченост.",
        "Анксиозност": "Анксиозност – зборови што изразуваат загриженост, страв или напнатост.",
        "Лутинa": "Лутинa – зборови што пренесуваат гнев, фрустрација или агресија.",
        "Тага": "Тага – зборови што одразуваат жалост, депресија или меланхолија.",
        "Пцовки": "Пцовки – вулгарни или непристојни зборови.",
        "Позитивен тон": "Позитивен тон – јазик кој создава пријатна и охрабрувачка атмосфера.",
        "Негативен тон": "Негативен тон – јазик кој создава непријатна или критична атмосфера.",
        "Неутрален тон": "Неутрален тон – јазик кој е рамномерен и без силна емоција.",
        "Зборови поврзани со аудитивна перцепција": "Зборови поврзани со аудитивна перцепција – зборови кои опишуваат слушање и звуци.",
        "Зборови поврзани со визуелна перцепција": "Зборови поврзани со визуелна перцепција – зборови кои опишуваат гледање и слики.",
        "Зборови поврзани со просторна перцепција": "Зборови поврзани со просторна перцепција – зборови што опишуваат позиција, насока и простор.",
        "Зборови поврзани со перцепција на движење": "Зборови поврзани со перцепција на движење – зборови кои опишуваат промени на позиција и динамика.",
        "Зборови поврзани со внимание": "Зборови поврзани со внимание – зборови кои опишуваат фокус, концентрација или забележување.",
        "Зборови поврзани со привлечност": "Зборови поврзани со привлечност – зборови кои изразуваат убавина, шарм или привлечност.",
        "Зборови поврзани со љубопитност": "Зборови поврзани со љубопитност – зборови што изразуваат интерес и желба за знаење.",
        "Зборови поврзани со ризик": "Зборови поврзани со ризик – зборови што укажуваат на опасност или неизвесност.",
        "Зборови поврзани со награда": "Зборови поврзани со награда – зборови кои изразуваат добивка, успех или придобивка.",
        "Зборови кои изразуваат потреби": "Зборови кои изразуваат потреби – зборови кои опишуваат неопходни услови за преживување.",
        "Зборови кои изразуваат желби": "Зборови кои изразуваат желби – зборови кои укажуваат на лични стремежи или копнежи.",
        "Зборови кои изразуваат стекнување": "Зборови кои изразуваат стекнување – зборови што означуваат добивање или поседување.",
        "Зборови кои изразуваат недостаток": "Зборови кои изразуваат недостаток – зборови што означуваат губење или отсуство.",
        "Зборови кои изразуваат исполнување": "Зборови кои изразуваат исполнување – зборови кои означуваат постигнување или задоволување.",
        "Зборови кои изразуваат замор": "Зборови кои изразуваат замор – зборови кои укажуваат на умор или исцрпеност.",
        "Зборови кои изразуваат болест": "Зборови кои изразуваат болест – зборови што означуваат физичка или психичка болест.",
        "Зборови кои изразуваат здравје": "Зборови кои изразуваат здравје – зборови што означуваат благосостојба и виталност.",
        "Зборови поврзани со ментално здравје": "Зборови поврзани со ментално здравје – зборови што укажуваат на психолошка состојба.",
        "Зборови поврзани со храна или јадење": "Зборови поврзани со храна или јадење – зборови што опишуваат исхрана или акти на јадење.",
        "Зборови поврзани со смрт": "Зборови поврзани со смрт – зборови што опишуваат умирање, крај или загуба.",
        "Зборови поврзани со самоповредување": "Зборови поврзани со самоповредување – зборови што опишуваат нанесување штета на себеси.",
        "Сексуална содржина": "Сексуална содржина – зборови што се однесуваат на секс или еротика.",
        "Зборови поврзани со слободно време": "Зборови поврзани со слободно време – зборови што опишуваат рекреација и хоби.",
        "Зборови поврзани со дом": "Зборови поврзани со дом – зборови што се однесуваат на живеалиште и семејство.",
        "Зборови поврзани со работа": "Зборови поврзани со работа – зборови што се однесуваат на професија и задачи.",
        "Зборови поврзани со пари": "Зборови поврзани со пари – зборови што опишуваат финансии и економија.",
        "Зборови поврзани со религија": "Зборови поврзани со религија – зборови што опишуваат духовност и верски практики.",
        "Зборови поврзани со политика": "Зборови поврзани со политика – зборови што опишуваат власт и општествени односи.",
        "Зборови поврзани со култура": "Зборови поврзани со култура – зборови што се однесуваат на уметност, традиции и идентитет.",
        "Пцовки": "Пцовки – повторно вулгарни или непристојни зборови.",
        "Странски зборови": "Странски зборови – зборови кои не припаѓаат на македонскиот јазик.",
        "Научни зборови": "Научни зборови – термини кои се користат во академски или стручни контексти.",
        "Сленг зборови": "Сленг зборови – неформални и разговорни изрази.",
        "Сленг од социјални мрежи": "Сленг од социјални мрежи – современ интернет сленг и кратенки.",
        "Зборови-полнители (filler words)": "Зборови-полнители (filler words) – зборови кои не додаваат значење, но ја пополнуваат реченицата.",
        "Зборови фокусирани на минатото": "Зборови фокусирани на минатото – изрази што се однесуваат на претходни настани.",
        "Зборови фокусирани на сегашноста": "Зборови фокусирани на сегашноста – изрази што се однесуваат на моменталното време.",
        "Зборови фокусирани на иднината": "Зборови фокусирани на иднината – зборови што се однесуваат на претстојни настани.",
        "Зборови поврзани со време": "Зборови поврзани со време – зборови што опишуваат периоди и траење.",
        "Погрешно напишани зборови": "Погрешно напишани зборови – зборови со правописни грешки.",
        "Повторени зборови": "Повторени зборови – зборови кои се појавуваат повеќепати без причина.",
        "Зборови кои изразуваат количина": "Зборови кои изразуваат количина – зборови што означуваат број или мера.",
        "Зборови кои означуваат семејство": "Зборови кои означуваат семејство – зборови што именуваат членови на семејство.",
        "Зборови кои означуваат пријатели": "Зборови кои означуваат пријатели – зборови што се однесуваат на пријателски односи.",
        "Зборови кои означуваат мажи": "Зборови кои означуваат мажи – зборови што се однесуваат на машки личности.",
        "Зборови кои означуваат жени": "Зборови кои означуваат жени – зборови што се однесуваат на женски личности.",
        "Зборови кои означуваат миленици": "Зборови кои означуваат миленици – зборови што именуваат домашни животни.",
        "Зборови кои означуваат социјален статус": "Зборови кои означуваат социјален статус – зборови што покажуваат општествена положба.",
        "Зборови кои означуваат сиромаштија": "Зборови кои означуваат сиромаштија – зборови што укажуваат на недостиг на средства.",
        "Зборови кои означуваат богатство": "Зборови кои означуваат богатство – зборови што укажуваат на изобилство или финансиска моќ.",
        "Интерпункциски симболи": "Интерпункциски симболи – сите знаци на интерпункција (.,!?).",
        "Составени зборови со цртичка": "Составени зборови со цртичка – зборови што содржат тире (на пр. црвено-бел).",
        "Оксфордска запирка": "Оксфордска запирка – запирка пред 'и' во список.",
        "Зборови во загради": "Зборови во загради – дополнителен текст вметнат меѓу загради.",
        "Броеви": "Броеви – зборови или симболи што означуваат количества.",
        "Издолжени зборови": "Издолжени зборови – зборови со намерно продолжени букви (на пр. „долгоооо“)."
    }

        self.system={"role": "system","content": 
            ("[INST]Разговор помеѓу корисник и разговорник за екстракција на стил македонска поезија. Асистентот дава корисни, детални и љубезни одговори на прашањата на корисникот.Ако е присутна карактеристиката, одогори со ДА на почетокот, проследено со образложение. Ако не е присутна, одговори само со НЕ и ништо друго.Оцени го присуството од 0 до 1 на стилот.[/INST]</s>.")}
        
        self.db=PoetryDB()
        self.CSV_PATH="classification/cleaned_songs.csv"
        self.df=pd.read_csv(self.CSV_PATH)
        self.random_seed=47
        self.styles_path='extracted_styles.csv'
        self.bedrock = boto3.client("bedrock-runtime", region_name="eu-central-1")
        
        
    def extract_n_random_songs_for_author(self, author_name='Блаже Конески', number_of_songs=10):
        author_songs = self.df[self.df['author'] == author_name]
        return author_songs.sample(
            n=min(number_of_songs, len(author_songs)),
            random_state=None  
        )
    def extract_all_songs_for_author(self, author_name='Блаже Конески'):
        return self.df[self.df['author'] == author_name]
    
    def extract_styles_from_song(self,song):
        pass
    def extract_style_from_song(self,song,target_feature,target_feature_definition):
        
        user_message = (
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
        messages = [self.system,{"role": "user", "content": user_message}]

        payload = {
        "model": "trajkovnikola/MKLLM-7B-Instruct",
        "messages": messages,
        "temperature": 0.3,
        "repetition_penalty":2,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.3,
        "top_p": 0.9,
        "max_tokens":200,
        "stop": ["\n","\n\n","<|im_end|>"]}
        
        resp = requests.post("http://127.0.0.1:8080/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload)
        
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

            for category, definition in self.styles_map.items():
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
    def apply_styles_iterative(self, styles, st_song_text, st_song_title, st_author, log_path=None):
    
        song = st_song_text

        
        if log_path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"song_style_log_{st_song_title or 'song'}_{ts}.txt"
            log_path = "".join(c for c in log_path if c.isalnum() or c in (' ','.','_','-')).rstrip()

        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        cumulative = 0.0
        total_styles = len(styles)

        with open(log_path, "a", encoding="utf-8") as log_file:
            for i, (_, row) in enumerate(styles.iterrows()):
                target_feature = row['style_feature_category']
                target_feature_definition = self.styles_map.get(target_feature, "")
                
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
                    "model": "trajkovnikola/MKLLM-7B-Instruct",
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
    def apply_styles_all_at_once(self,styles,st_song_text):
           
            song=st_song_text
            max_words=len(st_song_text)
           
            target_features = []
            target_feature_definitions = []

            for _, row in styles.iterrows():
                target_feature = row['style_feature_category']
                target_features.append(target_feature)
                target_feature_definitions.append(self.styles_map[target_feature])

            # Join definitions in a readable numbered format
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

            user_message = (
                f"{definitions_text}\n"
                f"Ова ја претставува дефиницијата, не ја давај нејзе, во твојот одговор.\n\n"
                f"Искористи ја оваа особина {target_feature} ВРЗ песната.\n"
                f"Како одговор врати ја назад песната, но со применет {target_feature} врз нејзе. Оваа е клучно.\n"
                f"Следуваат два примери:\n"
                f"{example_1}"
                f"{example_2}"
                f"Пасус:\n{song}\n\nОбработена песна:"
            )
               
            new_system={"role": "system","content": 
                ("[INST]Разговор помеѓу корисник и разговорник  за применување на  стил македонска поезија. Асистентот дава корисни, детални и љубезни одговори на прашањата на корисникот.[/INST]</s>.")}
            
            messages = [new_system,{"role": "user", "content": user_message}]

            payload = {
                "model": "trajkovnikola/MKLLM-7B-Instruct",
                "messages": messages,
                "temperature": 0.2,
                "repetition_penalty": 2.0,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.15,
                "top_p": 0.9,
                "max_tokens":max_words,
                'stop':'<|im_end|>"'}
            start=time.time()   
            resp = requests.post("http://127.0.0.1:8080/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload) 
            print(f'Time needed {time.time()-start}') 
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        
    def transfer_style(self,sf_author,sf_song_title,st_author,st_song_text,st_song_title): 
        #self,sf_author,sf_song_title):
        selected=self.get_present_styles_for_song(sf_song_title,sf_author)
        print(selected)
        return self.apply_styles_all_at_once(styles=selected,st_song_text=st_song_text)
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

        total_items = len(sample_songs) * len(self.styles_map)
        pbar = tqdm(total=total_items, desc="Extracting styles", ncols=100)

        for _, row in sample_songs.iterrows():
            author = row["author"]
            song = row["song_title"]
            

            for category, definition in self.styles_map.items():
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
            
test=StyleTransfer()


test.extract_style_from_all_songs(
    songs_csv="classification/cleaned_songs.csv",
    save_every=20  
)
 