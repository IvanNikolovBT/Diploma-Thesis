
import re
import sys
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from pprint import pprint
import wikipedia

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

        infobox = {}
        tbl = soup.find("table", class_="infobox")
        if tbl:
            for row in tbl.find_all("tr"):
                th, td = row.find("th"), row.find("td")
                if th and td:
                    key = th.get_text(" ", strip=True)
                    val = " ".join(td.stripped_strings)
                    infobox[key] = val

        result = {
            "page_title": title,
            "full_text": full_text,
            "infobox": infobox,
            "link":url
        }
            
        return result
    def fill_missing_biographies(self):
        authors=self.db.get_all_authors()
        for author in authors:
            result=self.scrape(author)
            if result is None:
                print(f' Couldnt find author {author} on Wikipedia')
            else:
                author_id=self.db.get_author_id(author)
                self.db.insert_biography(author_id,result["full_text"],result["link"])
                print(f'Succesfully updated author {author} with id {author_id}')
                print(f"Succesfuly inserted the biography")
                
if __name__ == "__main__":
    scraper = WikipediaScraper()
    scraper.fill_missing_biographies()
    
