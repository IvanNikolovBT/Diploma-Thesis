import time
import requests
from bs4 import BeautifulSoup
import re,sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from poetry_DB import PoetryDB


class KafeIKnigiScraper:
    BASE_URL = "https://kafeiknigi.com/category/%D0%BF%D0%BE%D0%B5%D0%BC%D0%B8-2/"

    def __init__(self, delay: float = 0.5):
        """
        :param delay: delay in seconds between page fetches
        """
        self.delay = delay
        self.all_links = []
        self.db = PoetryDB()
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
        Scrape all paginated links until no more are found,
        then save them to 'all_links.txt'.
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

        
        with open("all_links.txt", "w", encoding="utf-8") as f:
            for link in self.all_links:
                f.write(link + "\n")

        print(f"Saved {len(self.all_links)} unique links to 'all_links.txt'.")

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

    def extract_info(self, url: str):
        """
        Fetch HTML from URL and extract song info.
        """
        response = requests.get(url)
        response.raise_for_status()
        html = response.text

        soup = BeautifulSoup(html, 'html.parser')
        title_tag = soup.find('h1', class_="entry-title")
        title = title_tag.get_text(strip=True) if title_tag else "N/A"
      

        pattern = r'Македонска поезија: „(.+?)“ од ([\w\s]+)'
        matches = re.match(pattern, title)
        if not matches:
            print(f"[WARNING] Title pattern didn't match for URL {url}, skipping.")
            return
        
        song_name, author = matches[1], matches[2]
        content = soup.find('div', 'entry-content')
        context = content.find("p").get_text(strip=True) if content else None
        paragraphs = content.find_all("p")[1:] if content else []

        song = ""
        for p in paragraphs:
            text = p.get_text(separator="\n", strip=True)
            song += text + "\n\n"

        self.db.insert_kik_song(author, song_name, context, song)
    def scrape_and_extract_all(self):
        """
        Read links from file and extract info from each URL.
        """
        links_file = "/home/ivan/Desktop/Diplomska/scrapers/kafeiknigi/all_links.txt"
        
        with open(links_file, "r", encoding="utf-8") as f:
            self.all_links = [line.strip() for line in f if line.strip()]

        for i, link in enumerate(self.all_links, start=1):
            print(f"Processing link {i}/{len(self.all_links)}: {link}")
            try:
                self.extract_info(link)
            except Exception as e:
                print(f"[ERROR] Failed to extract info from {link}: {e}")
            time.sleep(self.delay)
            
        
test=KafeIKnigiScraper()

test.scrape_and_extract_all()



      

