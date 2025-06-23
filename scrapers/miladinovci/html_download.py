from bs4 import BeautifulSoup

html = open('scrapers/miladinovci/Ѕвездена_терезија.html')

soup = BeautifulSoup(html, "html.parser")

element_set = soup.find("div", class_="element-set")

elements = element_set.find_all("div", class_="element")

for element in elements:
    label = element.find("h3").get_text(strip=True)
    value = element.find("div", class_="element-text").get_text(strip=True)
    print(f"{label}: {value}")
    
img = soup.find("img", class_="BRpageimage BRnoselect")
src = img["src"]
print(src)    