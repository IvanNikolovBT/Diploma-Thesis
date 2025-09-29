import requests
import time
import subprocess
import json
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
    """

system = {
    "role": "system",
    "content": "Ти си македонски разговорник кој анализира стил и граматика на песни поезија."
}

u1 = {
    "role": "user",
    "content": (
        f"Напиши долг пасус кој ја опишува специфичната граматичка структура на поезијата.\n\n"
        f"Строго задржи се само за граматиката и нејзината анализа."
        f"Поезијата:\n{poem.strip()}\n\nОпис:"
    )
}

messages = [system, u1]

payload = {
    "model": "mistral-mk",
    "messages": messages,
    "temperature": 0.2,       
    "frequency_penalty": 0.8,   
    "presence_penalty": 0.5     
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
reply = data["choices"][0]["message"]["content"].strip()

style_description = reply

print("Extracted Style Description:\n", style_description)
print(f"Time {elapsed_1:.2f}s  CPU {cpu0}→{cpu1}°C  GPU {gpu0}→{gpu1}°C")

with open("response_log.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps({
        "poem": poem[:50],
        "style_description": style_description,
        "elapsed": elapsed_1,
        "cpu0": cpu0, "cpu1": cpu1,
        "gpu0": gpu0, "gpu1": gpu1,
        "ts": time.time()
    }, ensure_ascii=False) + "\n")

system_2 = {
    "role": "system",
    "content": "Ти си македонски разговорник кој анализира стил и граматика на песни поезија."
}

u_2 = {
    "role": "user",
    "content": (
        f"Ова е опис на стилот на авторот за дадената поезија. "
        f"Претвори го во листа од 5–6 кратки реченици, од форматот "
        f"'Авторот е X' или 'Авторот користи X' каде што X e својство поврзано со граматика. "
        f"Напрати скратена верзија на анализата, држти се за суштината.\n\n"
        f"Пример: Авторот користи минато свршено време."
        f"Пример: Авторот зборува со трето лице л форма.n\nn\n"
        f"Долгиот одвоор што треба да се скрати:\n{reply}\n\n"
    )
}

messages_2 = [system_2, u_2]
payload_2 = {
    "model": "mistral-mk",
    "messages": messages_2,
    "temperature": 0.2,
    "frequency_penalty": 0.9,
    "presence_penalty": 0.7,
}
start_2= time.time()
resp = requests.post(
    "http://127.0.0.1:8080/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json=payload_2
)

elapsed_2 = time.time() - start_2
cpu1 = get_cpu_temp()
gpu1 = get_gpu_temp()

if resp.status_code != 200:
    print("Error", resp.status_code, resp.text)
    sys.exit(1)

data = resp.json()
reply = data["choices"][0]["message"]["content"].strip()


style_description = reply
print("Extracted Style Description 2 :\n", style_description)
print(f"Time {elapsed_2:.2f}s  CPU {cpu0}→{cpu1}°C  GPU {gpu0}→{gpu1}°C")
print(f'Total time {elapsed_2+elapsed_1}')