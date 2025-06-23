
from PIL import Image
import pytesseract
from poetry_DB import PoetryDB


class ExtractorPipeline:
    
    def __init__(self):
        self.db=PoetryDB()
        
    def extract_test_raw(self,image_path:str)->str:
        img = Image.open(image_path)
        raw = pytesseract.image_to_string(img, lang='mkd')        
        return raw