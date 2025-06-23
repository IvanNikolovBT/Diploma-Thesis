import requests
import time
import subprocess
import json
import re
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
            ["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader"],
            capture_output=True, text=True
        ).stdout.strip()
    except:
        return "N/A"

if len(sys.argv) > 1:
    poem = sys.argv[1]
else:
    poem = """
    Првите десет инјекции

    Имам седум години и висока температура
    чекаме пред белата врата

    ...
    ми купи портокалово зајче,
    кое потем секаде го влечкам со себе.
    """
    
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
    "content": (
        "Ти си Македонски четбот за извлекување наслови. "
        "Одговараш со точниот наслов на песната и ништо повеќе."
    )
}

u1 = {"role": "user", "content": "Песна:\nРозе и сјај\nСе кревам на утринско светло.\nНаслов:"}
a1 = {"role": "assistant", "content": "Розе и сјај"}

u2 = {"role": "user", "content": "Песна:\nГрад на солзи\nКамени улици и сиви сенки.\nНаслов:"}
a2 = {"role": "assistant", "content": "Град на солзи"}


u_target = {
    "role": "user",
    "content": f"Песна:\n{poem.strip()}\nНаслов:"
}

messages = [system, u1, a1, u2, a2, u_target]

payload = {
    "model": "mistral-mk",
    "messages": messages,
    "max_tokens": 100,
    "temperature": 0.0,
    "stop": ["\n",'<|im_end|>']
}

start = time.time()
cpu0 = get_cpu_temp()
gpu0 = get_gpu_temp()

resp = requests.post(
    "http://127.0.0.1:8080/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json=payload
)

elapsed = time.time() - start
cpu1 = get_cpu_temp()
gpu1 = get_gpu_temp()

if resp.status_code != 200:
    print("Error", resp.status_code, resp.text)
    sys.exit(1)

data = resp.json()
reply = data["choices"][0]["message"]["content"].strip()

m = re.match(r"(.+)", reply)
title = m.group(1) if m else reply


print("Extracted Title:", title)
print(f"Time {elapsed:.2f}s  CPU {cpu0}→{cpu1}°C  GPU {gpu0}→{gpu1}°C")

with open("response_log.txt","a",encoding="utf-8") as f:
    f.write(json.dumps({
        "poem": poem[:50],
        "title": title,
        "elapsed": elapsed,
        "cpu0": cpu0,"cpu1": cpu1,
        "gpu0": gpu0,"gpu1": gpu1,
        "ts": time.time()
    }, ensure_ascii=False) + "\n")
