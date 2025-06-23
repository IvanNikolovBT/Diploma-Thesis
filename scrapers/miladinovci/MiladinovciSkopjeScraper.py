import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote
from typing import Optional, Dict
from urllib.parse import unquote, urlparse, parse_qs
import subprocess
import urllib.parse


class MiladinovciSkopjeScraper:
    BASE_URL: str = "https://digitalna.gbsk.mk"
    HEADERS: Dict[str, str] = {"User-Agent": "Mozilla/5.0"}

    def __init__(self, start_url: str, output_dir: str) -> None:
        self.start_url: str = start_url
        self.output_dir: str = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.book_id: int = int(re.search(r'/show/(\d+)', start_url).group(1))

    def fetch_soup(self, url: str) -> BeautifulSoup:
        r: requests.Response = requests.get(url, headers=self.HEADERS, timeout=10)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")

    def get_next_url(self, soup: BeautifulSoup) -> Optional[str]:
        nxt = soup.select_one("li.next a")
        return urljoin(self.BASE_URL, nxt["href"]) if nxt else None

    def find_book_info(self, soup: BeautifulSoup) -> Dict[str, str]:
        elements = soup.select("div.element-set div.element")
        info: Dict[str, str] = {
            el.find("h3").text.strip(): el.find("div", class_="element-text").text.strip()
            for el in elements
        }
        return info

    def get_image_url(self, page_index: int, scale: int = 1) -> str:
        image_code: str = f"{page_index + 1:04d}"
        return f"https://digitalna.gbsk.mk/book-reader/index/image-proxy/?image={image_code}&id={self.book_id}&scale={scale}"

    def download_image(self, url: str, path: str) -> bool:
        r: requests.Response = requests.get(url, headers=self.HEADERS, timeout=10)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"Downloaded {os.path.basename(path)}")
            return True
        print(f"Failed to download. {url}")
        return False

    def book_already_downloaded(self) -> bool:
        prefix = f"{self.book_id}"
        for filename in os.listdir(self.output_dir):
            if filename.startswith(prefix):
                print(f"Book {self.book_id} already downloaded, skipping...")
                return True
          
            
        return False

      
    

    def scrape(self, max_pages: int = 1000) -> None:
        url: Optional[str] = self.start_url
        poetry_pattern = re.compile(r"поезија|поема|песна|песни|стих|стихозбирка", re.IGNORECASE)

        while url:
            print(f"Processing: {url}")

            new_id_match = re.search(r'/show/(\d+)', url)
            if new_id_match:
                self.book_id = int(new_id_match.group(1))

            if self.book_already_downloaded():
                soup = self.fetch_soup(url)
                url = self.get_next_url(soup)
                continue

            soup: BeautifulSoup = self.fetch_soup(url)
            self.download_pdf_if_exists(soup)

            page_text = soup.get_text()
            if not poetry_pattern.search(page_text):
                print("Page does not contain poetry-related keywords, skipping...")
                url = self.get_next_url(soup)
                continue

            info: Dict[str, str] = self.find_book_info(soup)

            with open(os.path.join(self.output_dir, "scrape_log.txt"), "a", encoding="utf-8") as f:
                f.write("===========================================================================\n")
                f.write(f"Page link: {url}\n")
                f.write(f"Author: {info.get('Автор', 'N/A')}\n")
                f.write(f"Book: {info.get('Наслов', 'N/A')}\n")
                f.write(f"Object Label: {info.get('Предметна одредница', 'N/A')}\n")
                f.write(f"Language: {info.get('Јазик', 'N/A')}\n")
                f.write(f"Date: {info.get('Датум', 'N/A')}\n\n")

            print(", ".join(f"{k}: {v}" for k, v in info.items()))

            pages_downloaded = 0
            for i in range(max_pages):
                img_url: str = self.get_image_url(i)
                img_path: str = os.path.join(self.output_dir, f"{self.book_id}_page_{i+1:04}.jpg")
                if self.download_image(img_url, img_path):
                    pages_downloaded += 1
                else:
                    print(f"Stopped at page {i+1}, no more images.")
                    break

            print(f"Total pages downloaded for book {self.book_id}: {pages_downloaded}\n")

            url = self.get_next_url(soup)


    def download_pdf(self,viewer_url: str, output_filename: str, download_dir = '/home/ivan/Desktop/Diplomska/downloaded_pdfs') -> None:
        
        parsed = urllib.parse.urlparse(viewer_url)
        query_params = urllib.parse.parse_qs(parsed.query)
        encoded_pdf_url = query_params.get('url')
        if not encoded_pdf_url:
            raise ValueError("No 'url' parameter found in viewer URL")
        
        pdf_url = urllib.parse.unquote(encoded_pdf_url[0])
        
        os.makedirs(download_dir, exist_ok=True)
        
        
        output_path = os.path.join(download_dir, output_filename)
        
        shell_command = f'wget -O "{output_path}" "{pdf_url}"'
        
        result = subprocess.run(shell_command, shell=True, text=True, capture_output=True)
        if result.returncode != 0:
            print("Error downloading PDF:", result.stderr)
        else:
            print(f"Downloaded PDF saved as: {output_path}")
    def scrape_pdfs(self) -> None:
        url: Optional[str] = self.start_url

        while url:
            print(f"Processing (PDF only): {url}")

            new_id_match = re.search(r'/show/(\d+)', url)
            if new_id_match:
                self.book_id = int(new_id_match.group(1))
            
            if self.book_already_downloaded():
                soup = self.fetch_soup(url)
                url = self.get_next_url(soup)
                print()
                continue
            soup: BeautifulSoup = self.fetch_soup(url)
            
            js_iframe_string = (
                '// Set the default docviewer.\n'
                '    docviewer.append(\n'
                '  \'<iframe src="\' +'
            )
            soup_string=str(soup)
            start_index = soup_string.find(js_iframe_string)
            
            new_doc=soup_string[len(js_iframe_string)+start_index:]

            match = re.search(r'"([^"]*)"', new_doc)
            
            if len(match[0])!=2:
                url = match.group(1)
                pdf_url = 'https:' + url.replace('\\/', '/')
                print("Extracted URL:", pdf_url)
                self.download_pdf(pdf_url, f"{self.book_id }.pdf",'/home/ivan/Desktop/Diplomska/downloaded_books')
            else:
                print("No PDF URL found in the iframe.")       
            
           

            url = self.get_next_url(soup)

if __name__ == "__main__":
    scraper = MiladinovciSkopjeScraper("https://digitalna.gbsk.mk/items/show/9", "downloaded_books")
    scraper.scrape_pdfs()
    print("Scraping complete.")
