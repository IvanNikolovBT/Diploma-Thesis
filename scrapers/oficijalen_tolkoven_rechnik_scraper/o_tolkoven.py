import requests
from bs4 import BeautifulSoup
import re
import poetry_DB 
class TolkovenScraper:
    def __init__(self):
        self.session = requests.Session()
        self.db=poetry_DB.PoetryDB()
    def scrape_entry(self, url):
        try:
            response = self.session.get(url)
            response.raise_for_status()
            html = response.text

            soup = BeautifulSoup(html, "html.parser")
            section = soup.find("section")

            word_title_el = section.find("h1", class_="p-0 mb-5 font-weight-bold")
            word_title = word_title_el.get_text(strip=True) if word_title_el else None
            word_title = re.sub(r'\d+', '', word_title)
            pos_el = section.find("p", class_="p-0 m-0 mb-5")
            pos_tags = [a.get_text(strip=True) for a in pos_el.find_all("a")] if pos_el else []

            full_text = section.get_text(separator="\n", strip=True) if section else ""

            return {
                "title": word_title,
                "pos_tags": pos_tags,
                "full_text": full_text
            }

        except requests.exceptions.RequestException as e:
            print(f"Error fetching the page: {e}")
            return None
    
    def iterate_inside_of_letter(self, page_url): 
        try:
            response = self.session.get(page_url)
            response.raise_for_status()
            html = response.text

            soup = BeautifulSoup(html, "html.parser")

            base_url = "https://makedonski.gov.mk"

            
            links = soup.find_all('div', class_='content m-0 p-0 mb-10 pb-20')
            for entry in links:
                a_tag = entry.find('a')
                if a_tag and a_tag.get('href'):
                    entry_url = base_url + a_tag['href']
                    data = self.scrape_entry(entry_url)
                    title=data['title']
                    pos_tags=data['pos_tags']
                    full_text=data['full_text']
                    if data:
                        print("\n🔤 Title:",title)
                        print("🧩 POS Tags:", pos_tags)
                        print("📄 Full Text:\n", full_text)
                        print("=" * 40)
                        self.db.insert_word_information_o_tolkoven(title,pos_tags,full_text)
            return False

        except requests.exceptions.RequestException as e:
            print(f"Error fetching the page: {e}")
            return True
    def get_letter_info(self,letter):
        base_url=f'https://makedonski.gov.mk/bukva/{letter}'
        i=2

        while True:
            
            current_url=base_url+f'?strana={i}'
            res=self.iterate_inside_of_letter(current_url)
            i+=1
            if res:
                break
            
            
    def rotate_all_letters(self):
        macedonian_alphabet = [
            'а', 'б', 'в', 'г', 'д', 'ѓ', 'е', 'ж', 'з', 'ѕ', 'и', 'ј',
            'к', 'л', 'љ', 'м', 'н', 'њ', 'о', 'п', 'р', 'с', 'т', 'ќ',
            'у', 'ф', 'х', 'ц', 'ч', 'џ', 'ш'
        ]
        macedonian_alphabet = [
            'п', 'р', 'с', 'т', 'ќ',
            'у', 'ф', 'х', 'ц', 'ч', 'џ', 'ш'
        ]
       
        for letter in macedonian_alphabet:
            self.get_letter_info(letter)         
url='https://makedonski.gov.mk/corpus/l/a-2-svrz'        
scraper = TolkovenScraper()
#data = scraper.scrape_entry(url)
page_url='https://makedonski.gov.mk/bukva/%D0%B0'
page_url='https://makedonski.gov.mk/bukva/%D0%B1?strana=1'
#scraper.get_letter_info('б')
scraper.rotate_all_letters()