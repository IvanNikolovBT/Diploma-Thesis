import requests
import time
import subprocess
import json


def get_cpu_temp():
    try:
        result = subprocess.run(["sensors"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "Core 0" in line: 
                return line.split()[2].strip("+°C")
        return "N/A"
    except:
        return "N/A"


def get_gpu_temp():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "N/A"


start = time.time()
initial_cpu_temp = get_cpu_temp()
initial_gpu_temp = get_gpu_temp()


response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    json={
        "messages": [
            {
                "role": "system",
                "content": "Ти си Македонски четбот кој дава кратки, точни и проверени одговори на македонски јазик. Одговарај само на поставеното прашање, избегнувај повторувања, измислици или додавање на непотребни детали."
            },
            {
                "role": "user",
                "content": "Кој е Бранислав Николов"
            }
        ],
        "max_tokens": 500,  
        "temperature": 0.1,  
        "stop": ["<|im_end|>"],  
        "repetition_penalty": 1.2,
        "repeat_last_n":64
    }
)


end = time.time()
elapsed = end - start
final_cpu_temp = get_cpu_temp()
final_gpu_temp = get_gpu_temp()


if response.status_code != 200:
    print(f"Error: {response.status_code}")
    print(response.json())
    exit(1)


response_json = response.json()
full_response = response_json["choices"][0]["message"]["content"]
relevant_response = full_response.split("<|im_end|>")[0].strip()


def validate_response(response_text, question):
    if "Битола" in question and "трет по големина" in response_text:
        return "Битола е втор по големина град во Македонија."
    return response_text

validated_response = validate_response(relevant_response, "Што е Битола?")


print("=== Raw Response ===")
print(response.text)

print("=== Model Response ===")
print(validated_response)

print("\n=== Timings (From Server) ===")
print(f"Prompt Time: {response_json['timings']['prompt_ms'] / 1000:.2f} sec")
print(f"Generation Time: {response_json['timings']['predicted_ms'] / 1000:.2f} sec")
print(f"Total Server Time: {(response_json['timings']['prompt_ms'] + response_json['timings']['predicted_ms']) / 1000:.2f} sec")

print("\n=== Total Wall-Clock Time (Client-side) ===")
print(f"Elapsed Time: {elapsed:.2f} sec")

print("\n=== System Metrics ===")
print(f"Initial CPU Temp: {initial_cpu_temp}°C")
print(f"Final CPU Temp: {final_cpu_temp}°C")
print(f"Initial GPU Temp: {initial_gpu_temp}°C")
print(f"Final GPU Temp: {final_gpu_temp}°C")


with open("response_log.txt", "a") as log_file:
    log_entry = {
        "question": "Што е Битола?",
        "response": validated_response,
        "timestamp": time.time(),
        "cpu_temp_initial": initial_cpu_temp,
        "cpu_temp_final": final_cpu_temp,
        "gpu_temp_initial": initial_gpu_temp,
        "gpu_temp_final": final_gpu_temp
    }
    log_file.write(json.dumps(log_entry) + "\n")