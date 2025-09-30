import requests
import time
import subprocess
import sys
import threading

elapsed = 0
stop_timer = False

def show_timer():
    global elapsed, stop_timer
    while not stop_timer:
        print(f"\rWaiting: {elapsed:.1f}s", end="", flush=True)
        time.sleep(0.1)
        elapsed += 0.1
    print()
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
        "<s>[INST]Разговор помеѓу љубопитен корисник и разговорник за екстракција на стил македонска поезија. Асистентот дава корисни, детални и љубезни одговори на прашањата на корисникот.[/INST]</s>."
    )
}




user_message = (
    f"Извлечи ги најбитните зборови кои што се присутни во поезијата, според тебе. Извлечи ги како листа"
    f"Пасус:\n{poem}\n\nОпис:"
)

messages = [
    system,
    {"role": "user", "content": user_message}
]

payload = {
    "model": "trajkovnikola/MKLLM-7B-Instruct",
    "messages": messages,
    "temperature": 0.3,
    "repetition_penalty": 0.6,
    "frequency_penalty": 0.4,
    "presence_penalty": 0.3,
    "top_p": 0.9,
    "max_tokens":500,
    "stop": ["\n\n", "<|im_end|>"] 
}

timer_thread = threading.Thread(target=show_timer)
timer_thread.start()
cpu0 = get_cpu_temp()
gpu0 = get_gpu_temp()
start_time = time.time()
resp = requests.post(
    "http://127.0.0.1:8080/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json=payload
)

cpu1 = get_cpu_temp()
gpu1 = get_gpu_temp()

if resp.status_code != 200:
    print("Error", resp.status_code, resp.text)
    sys.exit(1)

total_time = time.time() - start_time
stop_timer = True  # Stop the timer
timer_thread.join()
data = resp.json()
style_description = data["choices"][0]["message"]["content"].strip()



print("Extracted Style Description:\n", style_description)
print(f"Time {total_time:.2f}s  CPU {cpu0}→{cpu1}°C  GPU {gpu0}→{gpu1}°C")


