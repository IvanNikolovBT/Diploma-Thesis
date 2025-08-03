import os
import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import unicodedata

from langchain.text_splitter import RecursiveCharacterTextSplitter


OCR_DPI = 300
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

class Preprocessor:
    def __init__(self,
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP,
                 ocr_if_needed: bool = True):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.ocr_if_needed = ocr_if_needed

    def load_txt(self,directory_path = "extracted_books_from_images"):
  
        file_contents = {}
        directory = Path(directory_path)
        for file_path in directory.glob('*.txt'):
            try:
                file_contents[file_path.name] = file_path.read_text(encoding='utf-8')
            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")
        return file_contents
    
processor=Preprocessor()
contents=processor.load_txt()
print(contents.keys())
       