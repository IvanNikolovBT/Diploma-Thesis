import sys
import os
import requests
from urllib.parse import quote
from pprint import pprint
import wikipedia
from typing import Optional, List, Dict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

wikipedia.set_lang("mk")
from poetry_DB import PoetryDB
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
# Optional helpers
from unidecode import unidecode
try:
    import mwparserfromhell
    _HAS_MWPARSER = True
except Exception:
    _HAS_MWPARSER = False



WIKI_API = "https://mk.wikipedia.org/w/api.php"
WIKI_BASE = "https://mk.wikipedia.org/wiki/"
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{}.json"


def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "WikipediaScraper/1.0 (you@example.com)"})
    # lightweight retry strategy
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
    retries = Retry(total=3, backoff_factor=0.2, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

_session = make_session()
_wikidata_cache: Dict[str, dict] = {}


class WikipediaScraper:
    def __init__(self):
        self.db = PoetryDB()
        self.session = _session

    def api_get(self, params: dict, timeout: float = 10.0) -> dict:
        r = self.session.get(WIKI_API, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def try_exact_title(self, name: str) -> Optional[str]:
        params = {"action": "query", "format": "json", "titles": name, "redirects": 1}
        j = self.api_get(params)
        pages = j.get("query", {}).get("pages", {})
        for pid, page in pages.items():
            if int(pid) > 0 and not page.get("missing", False):
                return page.get("title")
        return None

    def prefix_search(self, name: str, limit: int = 8) -> List[str]:
        params = {"action": "query", "format": "json", "list": "prefixsearch", "pssearch": name, "pslimit": limit}
        j = self.api_get(params)
        hits = j.get("query", {}).get("prefixsearch", [])
        return [h["title"] for h in hits]

    def api_search(self, name: str, limit: int = 8) -> List[str]:
        params = {"action": "query", "format": "json", "list": "search", "srsearch": name, "srlimit": limit}
        j = self.api_get(params)
        hits = j.get("query", {}).get("search", [])
        return [h["title"] for h in hits]

    def get_page_pageprops(self, title: str) -> dict:
        params = {"action": "query", "format": "json", "prop": "pageprops|categories", "titles": title, "redirects": 1, "ppprop": "wikibase_item", "cllimit": 50}
        j = self.api_get(params)
        pages = j.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        return page

    def get_wikitext(self, title: str) -> str:
        params = {"action": "query", "format": "json", "prop": "revisions", "rvprop": "content", "rvslots": "*", "titles": title, "redirects": 1}
        j = self.api_get(params)
        pages = j.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        revs = page.get("revisions", [])
        if not revs:
            return ""
        wikitext = revs[0].get("slots", {}).get("main", {}).get("*", "")
        return wikitext or ""

    def page_has_infobox(self, title: str) -> bool:
        # if mwparserfromhell available, parse templates; otherwise do cheap string check
        try:
            wikitext = self.get_wikitext(title)
            if not wikitext:
                return False
            if _HAS_MWPARSER:
                parsed = mwparserfromhell.parse(wikitext)
                for t in parsed.filter_templates():
                    name = str(t.name).lower()
                    if "infobox" in name:
                        return True
                return False
            else:
                # cheap fallback: look for common infobox start
                return "{{Infobox" in wikitext or "{{infobox" in wikitext
        except Exception:
            return False

    def wikidata_entity_is_human(self, qid: str) -> bool:
        if not qid:
            return False
        if qid in _wikidata_cache:
            ent = _wikidata_cache[qid]
        else:
            url = WIKIDATA_ENTITY_URL.format(qid)
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
            j = r.json()
            ent = j.get("entities", {}).get(qid, {})
            _wikidata_cache[qid] = ent
        claims = ent.get("claims", {})
        p31 = claims.get("P31", [])
        for claim in p31:
            dv = claim.get("mainsnak", {}).get("datavalue", {})
            val = dv.get("value", {})
            if isinstance(val, dict) and val.get("id") == "Q5":
                return True
        return False

    def get_page_extract(self, title: str) -> dict:
        params = {"action": "query", "format": "json", "prop": "extracts|pageprops", "titles": title, "explaintext": 1, "redirects": 1, "exsectionformat": "plain", "ppprop": "wikibase_item"}
        j = self.api_get(params)
        pages = j.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        extract = page.get("extract", "")
        pageprops = page.get("pageprops", {})
        return {"title": page.get("title"), "extract": extract, "pageprops": pageprops, "full_page": page}

    def normalize_str(self, s: str) -> str:
        try:
            return unidecode(s).strip().lower()
        except Exception:
            return s.strip().lower()

    def find_mk_title_strict(self, name: str) -> Optional[str]:
        
        t = self.try_exact_title(name)
        if t:
            page = self.get_page_pageprops(t)
            qid = page.get("pageprops", {}).get("wikibase_item")
            if qid and self.wikidata_entity_is_human(qid):
                return t
            if self.page_has_infobox(t):
                return t
            if self.normalize_str(t) == self.normalize_str(name):
                return t
        
        variants = [name, unidecode(name)]
        parts = name.split()
        if len(parts) >= 2:
            variants.append(" ".join([parts[-1]] + parts[:-1]))
            variants.append(parts[0] + " " + parts[-1])
        
        seen = set()
        variants = [v for v in variants if not (v in seen or seen.add(v))]

        candidate_titles: List[str] = []
        for v in variants:
            candidate_titles.extend(self.prefix_search(v, limit=8))
        if not candidate_titles:
            for v in variants:
                candidate_titles.extend(self.api_search(v, limit=8))
        
        seen = set()
        unique_candidates = []
        for c in candidate_titles:
            if c not in seen:
                unique_candidates.append(c)
                seen.add(c)

       
        best_score = -999
        best_title = None
        for title in unique_candidates:
            try:
                page = self.get_page_pageprops(title)
                qid = page.get("pageprops", {}).get("wikibase_item")
                score = 0
                if qid and self.wikidata_entity_is_human(qid):
                    score += 10
                if self.page_has_infobox(title):
                    score += 5
                if self.normalize_str(title) == self.normalize_str(name):
                    score += 3
                # intro proximity
                extract = self.get_page_extract(title).get("extract", "")
                if self.normalize_str(name) in self.normalize_str(extract[:300]):
                    score += 2
                if page.get("pageprops", {}).get("disambiguation"):
                    score -= 5
                if score > best_score:
                    best_score = score
                    best_title = title
                if score >= 15:
                    return title
            except Exception:
                continue
        return best_title

    def scrape(self, name: str) -> dict | None:
        title = self.find_mk_title_strict(name)
        if not title:
            print(f"No specific person page found for: {name}")
            return None
        page = self.get_page_extract(title)
        extract = page.get("extract", "")
        link = WIKI_BASE + quote(title.replace(" ", "_"))
        infobox_wikitext = self.get_wikitext(title) if _HAS_MWPARSER else None
        result = {
            "page_title": title,
            "full_text": extract,
            "infobox_wikitext": infobox_wikitext,
            "link": link,
            "pageprops": page.get("pageprops", {})
        }
        print(f"Found page: {title} for name: {name}")
        return result

    def fill_missing_biographies(self):
        authors = self.db.get_all_authors()
        for author in authors:
            result = self.scrape(author)
            if result is None:
                print(f"Couldn't find a specific person page for author: {author}")
            else:
                author_id = self.db.get_author_id(author)
                # guard: make sure full_text exists
                if not result.get("full_text"):
                    print(f"No extract for {author}, skipping insert")
                    continue
                self.db.insert_biography(author_id, result["full_text"], result["link"])
                print(f"Successfully updated author {author} with id {author_id}")


if __name__ == "__main__":
    scraper = WikipediaScraper()
    scraper.fill_missing_biographies()
