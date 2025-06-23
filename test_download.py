import requests

url = "https://digitalna.gbsk.mk/files/original/52cd9cebed292965186d49e550d72771.pdf"

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Accept": "application/pdf",
    "Referer": "https://digitalna.gbsk.mk/",
}

try:
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code == 200:
        with open("downloaded.pdf", "wb") as f:
            f.write(response.content)
        print("✅ Downloaded successfully.")
    else:
        print(f"❌ Server returned status code: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"⚠️ Request failed: {e}")
