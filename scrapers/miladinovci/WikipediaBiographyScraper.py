
import re
import sys
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from pprint import pprint


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from poetry_DB import PoetryDB

class WikipediaScraper:
    """
    A scraper for Macedonian Wikipedia biography pages.
    """
    WIKI_API = "https://mk.wikipedia.org/w/api.php"
    WIKI_BASE = "https://mk.wikipedia.org/wiki/"

    def __init__(self):
        self.db = PoetryDB()

    def find_mk_title(self, name: str) -> str | None:
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": name,
            "srlimit": 1
        }
        resp = requests.get(self.WIKI_API, params=params)
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        return results[0]["title"] if results else None

    def scrape(self, name: str) -> dict | None:
        title = self.find_mk_title(name)
        if not title:
            print(f"No Macedonian Wikipedia page found for '{name}'")
            return None

        url = self.WIKI_BASE + quote(title)
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        content_div = soup.find("div", class_="mw-parser-output")
        texts = []
        for tag in content_div.find_all(["h2", "h3", "h4", "p", "li"]):
            txt = tag.get_text(" ", strip=True)
            if txt:
                texts.append(txt)
        full_text = "\n".join(texts)
        first_para = texts[0] if texts else ""

        infobox = {}
        tbl = soup.find("table", class_="infobox")
        if tbl:
            for row in tbl.find_all("tr"):
                th, td = row.find("th"), row.find("td")
                if th and td:
                    key = th.get_text(" ", strip=True)
                    val = " ".join(td.stripped_strings)
                    infobox[key] = val

        dob = pob = pod = dod = gender = None
        if "Роден" in infobox:
            parts = [p.strip() for p in infobox["Роден"].split(",")]
            dob = parts[0]
            pob = parts[1] if len(parts) > 1 else None
        if "Починал" in infobox or "Умрел" in infobox:
            dod = infobox.get("Починал") or infobox.get("Умрел")
        if "Пол" in infobox:
            gender = infobox["Пол"]

        if not (dob and pob and dod):
            pattern_full = re.compile(
                r"\(\s*(?P<pob>.+?)\s*,\s*(?P<dob>\d{1,2}\s+\w+\s+\d{4})\s*—\s*(?P<pod>.+?)\s*,\s*(?P<dod>\d{1,2}\s+\w+\s+\d{4})\s*\)"
            )
            m = pattern_full.search(first_para)
            if m:
                pob = pob or m.group("pob").strip()
                dob = dob or m.group("dob").strip()
                pod = m.group("pod").strip()
                dod = dod or m.group("dod").strip()

        if not (dob and pob):
            pattern_birth_only = re.compile(
                r"\(\s*(?P<pob>.+?)\s*,\s*(?P<dob>\d{1,2}\s+\w+\s+\d{4})\s*\)"
            )
            m2 = pattern_birth_only.search(first_para)
            if m2:
                pob = pob or m2.group("pob").strip()
                dob = dob or m2.group("dob").strip()

        if not (dob and pob):
            male = re.search(
                r"Роден е во\s+(?P<loc>[^,]+),\s*во\s*(?P<year>\d{4})\s*година",
                full_text
            )
            female = re.search(
                r"Родена е во\s+(?P<loc>[^,]+),\s*во\s*(?P<year>\d{4})\s*година",
                full_text
            )
            if male:
                pob = pob or male.group("loc").strip()
                dob = dob or male.group("year").strip()
            elif female:
                pob = pob or female.group("loc").strip()
                dob = dob or female.group("year").strip()

        if not gender:
            if re.search(r"\bРоден е\b", full_text):
                gender = "Машко"
            elif re.search(r"\bРодена е\b", full_text):
                gender = "Женско"

        result = {
            "page_title": title,
            "full_text": full_text,
            "date_of_birth": dob,
            "place_of_birth": pob,
            "date_of_death": dod,
            "place_of_death": pod,
            "gender": gender,
            "infobox": infobox,
            "link":url
        }
            
        return result
    def fill_missing_biographies(self):
        
                
if __name__ == "__main__":
    scraper = WikipediaScraper()
    data = scraper.scrape("Петре М. Андреевски")
    pprint(data)
