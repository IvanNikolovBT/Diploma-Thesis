import time
import requests
from bs4 import BeautifulSoup
import re,sys,os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from poetry_DB import 
class KafeIKnigiScraper:
    BASE_URL = "https://kafeiknigi.com/category/%D0%BF%D0%BE%D0%B5%D0%BC%D0%B8-2/"

    def __init__(self, delay: float = 0.5):
        """
        :param delay: delay in seconds between page fetches
        """
        self.delay = delay
        self.all_links = []

    def _get_links_from_page(self, page_num: int) -> list[str]:
        """
        Fetch thumbnail links from a given category page.
        """
        if page_num == 1:
            url = self.BASE_URL
        else:
            url = self.BASE_URL + f"page/{page_num}/"

        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.text, "html.parser")
            return [a["href"] for a in soup.select("article a.entry-thumbnail")]
        except requests.RequestException:
            return []

    def scrape(self):
        """
        Scrape all paginated links until no more are found.
        """
        page = 1

        while True:
            print(f"Fetching page {page}…", end=" ")
            links = self._get_links_from_page(page)
            if not links:
                print("no more links, stopping.")
                break

            print(f"found {len(links)} links")
            self.all_links.extend(links)
            page += 1
            time.sleep(self.delay)

        
        self.all_links = list(dict.fromkeys(self.all_links))
    
    def get_links(self) -> list[str]:
        """
        Return the collected unique links.
        """
        return self.all_links

    def print_links(self):
        """
        Print total and all collected links.
        """
        print(f"\nTotal unique links collected: {len(self.all_links)}\n")
        for link in self.all_links:
            print(link)

    def extract_info(self,page_link:str):
        
        with open(page_link,"r",encoding='utf-8') as f: 
            html=f.read()
        soup=BeautifulSoup(html,'html.parser')
        title_tag=soup.find('h1',class_="entry-title")
        title = title_tag.get_text(strip=True) if title_tag else "N/A"
        print(title)
        pattern=r'Македонска поезија: „(.+)“ од ([А-Ша-шЃЌЏ\s]+)'
        matches=re.match(pattern,title)
        song_name,author=matches[1],matches[2]
        print(f'Song_name: {song_name}')
        print(f'Author: {author}')
        content=soup.find('div','entry-content')
        
        context = content.find("p").get_text(strip=True) if content else None
        print(f'Context: {context}')
        paragraphs = content.find_all("p")[1:] if content else []
        song=""
        for p in paragraphs:
            text = p.get_text(separator="\n", strip=True)
            song += text + "\n\n"  

        print(song.strip())
        

test=KafeIKnigiScraper()
test.extract_info("/home/ivan/Desktop/Diplomska/scrapers/kafeiknigi/template.html")       



      

