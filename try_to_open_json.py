import requests

url = "https://digitalna.gbsk.mk/items/show/1?output=json"
headers = {
    "User-Agent": "Mozilla/5.0"  # Pretend to be a browser
}

response = requests.get(url, headers=headers)
response.raise_for_status()  # This will throw if status is not 200
data = response.json()
print(data)
print("Title:", data.get("item_title"))
