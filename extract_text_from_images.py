
from PIL import Image

from poetry_DB import PoetryDB
import re,sys,os,pytesseract


class ExtractorPipeline:
    
    def __init__(self):
        self.db=PoetryDB()
        
    def extract_test_raw(self,image_path:str)->str:
        img = Image.open(image_path)
        raw = pytesseract.image_to_string(img, lang='mkd')        
        return raw
    def get_leading_ids(self,directory_path):
        ids = set()
        for entry in os.listdir(directory_path):
            if "_" in entry:
                leading = entry.split("_", 1)[0]
                if leading.isdigit():
                    ids.add(int(leading))
        return sorted(ids)
    def get_already_processed_book_titles(self,path:str):
        ids=set()
        for entry in os.listdir(path):
            ids.add(int(entry.split(".")[0]))
        return sorted(ids)
    
    def is_book_already_pressent(self,path:str,book_id):
        books_already_present=self.get_leading_ids(path)
        return book_id in books_already_present
    def extract_entire_book_from_images(self,input_path="/home/ivan/Desktop/Diplomska/downloaded_books",output_path="/home/ivan/Desktop/Diplomska/extracted_books_from_images"):
        
        books=self.get_leading_ids(input_path)
        present_books=self.get_already_processed_book_titles(output_path)
        
        for book_id in books:
            if book_id in present_books:
                print(f'Book with id {book_id} is already proccessed. Skipping it.')
                continue
            i=1
            file_path=f'{output_path}/{book_id}.txt'
            with open(file_path,'a') as f:
                while True:
                
                    image_path=f"{input_path}/{book_id}_page_{str(i).zfill(4)}.jpg" 
                    i+=1 
                    if not os.path.isfile(image_path):
                        break
                    raw_text=self.extract_test_raw(image_path)
                    print(raw_text)
                    f.write(raw_text)
    
                        
        
test=ExtractorPipeline()

test.extract_entire_book_from_images()
        