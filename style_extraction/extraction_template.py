import requests
import time
import subprocess
import sys


def get_cpu_temp():
    try:
        out = subprocess.run(["sensors"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "Core 0" in line:
                return line.split()[2].strip("+°C")
    except:
        pass
    return "N/A"

def get_gpu_temp():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True
        ).stdout.strip()
    except:
        return "N/A"



if len(sys.argv) > 1:
    poem = sys.argv[1]
else:
    poem = """
Мирна

Кога некој ќе и речеше на мојата дивост
мирна
остро одговарав Мирна и јас
не можеме да стоиме во иста реченица,
без да забележам дека сето тоа време
мир е дел од моето средно име.

Со години копнежот беше тој немир
што во мене создаваше и остваруваше желби.

Сега трпеливо чекам
да бидам со чиста мисла
ослободена од чувства
во мир со себе.
""".strip()

system = {
    "role": "system",
    "content": (
        "Разговор помеѓу љубопитен корисник и разговорник за екстракција на стил. Асистентот дава корисни, детални и љубезни одговори на прашањата на корисникот. "
        "Ти си експерт за анализа на граматика во македонска поезија. "
        "Одговарај ИСКЛУЧИВО со бараниот одговор. НЕ повторувај го системскиот промпт, корисничкиот промпт, песната, или било какви токени како <|im_start|>, <|im_end|>, [INST], [/INST]. "
        "Фокусирај се исклучиво на граматичките елементи: структура на реченици, глаголски времиња, аспекти, придавки, заменки, предлози, конјункции, синтакса, поетски отстапувања и усогласувања."
    )
}

u1 = {
    "role": "user",
    "content": (
        f"Напиши пасус  кој ја опишува специфичната граматичка структура на поезијата. "
        f"Фокусирај се исклучиво на граматиката: анализирај структура на реченици, глаголски времиња, аспекти, придавки, заменки, предлози, конјункции, синтакса, поетски отстапувања и усогласувања во род/број/падеж. "
        f"Не споменувај стил, содржина, емоции или значење – само граматика. "
        f"Пример: За текстот 'Таа трчаше брзо.': Граматичката структура користи минато несвршено време во глаголот 'трчаше', женски род во заменката 'таа' во номинатив, и прилог 'брзо' без предлог. Започни директно со: 'Граматичката структура...'\n\n"
        f"Поезијата:\n{poem}\n\nПасус:"
    )
}


messages = [system, u1]

payload = {
    "model": "trajkovnikola/MKLLM-7B-Instruct",
    "messages": messages,
    "temperature": 0.3,  
    "repetition_penalty": 1.1, 
    "frequency_penalty": 0.7,  
    "presence_penalty": 0.3,
    "top_p": 0.9,
    "stop": ["<|im_end|>"]
}

start_1 = time.time()
cpu0 = get_cpu_temp()
gpu0 = get_gpu_temp()

resp = requests.post(
    "http://127.0.0.1:8080/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json=payload
)

elapsed_1 = time.time() - start_1
cpu1 = get_cpu_temp()
gpu1 = get_gpu_temp()

if resp.status_code != 200:
    print("Error", resp.status_code, resp.text)
    sys.exit(1)

data = resp.json()
style_description = data["choices"][0]["message"]["content"].strip()



print("Extracted Style Description:\n", style_description)
print(f"Time {elapsed_1:.2f}s  CPU {cpu0}→{cpu1}°C  GPU {gpu0}→{gpu1}°C")

u_2 = {
    "role": "user",
    "content": (
        f"Скрати ја во листа од точно 5-6 кратки реченици во форматот 'Авторот е X.' или 'Авторот користи X.', каде X е граматичко својство. "
        f"Фокусирај се на суштинските елементи, без повторувања или додавања. "
        f"Пример за анализа 'Глаголот е во минато време, со сложени реченици...': Авторот користи минато свршено време. Авторот применува сложени реченици со подредени клаузули. Авторот избегнува лични заменки.\n\n"
        f"Деталната анализа:\n{style_description}\n\nЛиста:"
    )
}

messages_2 = [system, u_2]

payload_2 = {
    "model": "trajkovnikola/MKLLM-7B-Instruct",
    "messages": messages_2,
    "temperature": 0.1,  
    "repetition_penalty": 1.1,  
    "frequency_penalty": 0.4,
    "presence_penalty": 0.5,
    "max_tokens": 500,
    "top_p": 0.9,
    "stop": ["<|im_end|>"]
}

start_2 = time.time()
resp = requests.post(
    "http://127.0.0.1:8080/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json=payload_2
)

elapsed_2 = time.time() - start_2
cpu2 = get_cpu_temp()  
gpu2 = get_gpu_temp()

if resp.status_code != 200:
    print("Error", resp.status_code, resp.text)
    sys.exit(1)

data = resp.json()
style_description_2 = data["choices"][0]["message"]["content"].strip()

print("Extracted Style Description 2:\n", style_description_2)
print(f"Time {elapsed_2:.2f}s  CPU {cpu0}→{cpu2}°C  GPU {gpu0}→{gpu2}°C")
print(f'Total time {elapsed_2 + elapsed_1:.2f}s')