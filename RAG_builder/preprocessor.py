import os,sys
from pathlib import Path
import re
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import unicodedata
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from typing import List, Dict
import requests
import json
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_community.document_loaders import PDFMinerLoader


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from poetry_DB import PoetryDB
OCR_DPI = 300
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


class TextInfo:
    def __init__(self,
                 file_name,
                 book_title,
                 book_id,
                 author_full_name,
                 author_bio,
                 book_date,
                 book_object_label,text):
        self.file_name=file_name
        self.book_title=book_title
        self.book_id=book_id
        self.author_full_name=author_full_name
        self.author_bio=author_bio
        self.book_date=book_date
        self.book_object_label=book_object_label
        self.text=text

        
    def print_info(self):
        print("=== TextInfo ===")
        print(f"File name          : {self.file_name}")
        print(f"Book title         : {self.book_title}")
        print(f"Book ID            : {self.book_id}")
        print(f"Author full name   : {self.author_full_name}")
        print(f"Author bio         : {self.author_bio or '[none]'}")
        print(f"Book date          : {self.book_date}")
        print(f"Object label       : {self.book_object_label}")
        snippet = self.text.strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."
        print(f"Text snippet       : {snippet}")
        print("=================")

    def __str__(self):
        bio = self.author_bio if self.author_bio else "[none]"
        snippet = self.text.strip().replace("\n", " ")
        if len(snippet) > 100:
            snippet = snippet[:97] + "..."
       
        
        return (
            f"TextInfo(file_name={self.file_name}, book_title={self.book_title}, "
            f"book_id={self.book_id}, author={self.author_full_name}, "
            f"author_bio={bio}, book_date={self.book_date}, "
            f"object_label={self.book_object_label}, snippet={snippet})"
        )
    
    
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
        self.db=PoetryDB()

    
    def load_txt(self, directory_path: str = "extracted_books_from_images") -> Dict[str, Document]:

        """Load all .txt files into LangChain Documents with metadata."""
        file_documents = {}
        directory = Path(directory_path)
        pattern = re.compile(r'^(\d+)\.txt$')  # Matches '123.txt' and extracts '123' as book_id

        for file_path in directory.glob('*.txt'):
            try:
                # Read text content
                text = file_path.read_text(encoding='utf-8')

                # Extract book_id from filename (e.g., '123.txt' → '123')
                m = pattern.match(file_path.name)
                if not m:
                    print(f"Skipping '{file_path.name}': Filename pattern mismatch.")
                    continue

                book_id = m.group(1)
                book_info = self.db.get_book_info_by_id_in_link(book_id)

                # Create LangChain Document with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "file_name": file_path.name,
                        "book_id": book_info['book_id'],
                        "book_title": book_info['book_title'],
                        "author_full_name": book_info['full_name'],
                        "author_biography": book_info['author_biography'],
                        "date": book_info['date'],
                        "object_label": book_info['object_label']
                    }
                )
                
                file_documents[book_id] = doc

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

            print(f"Loaded {len(file_documents)} documents.")
            return file_documents
    

        docs: List[Document] = []
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if not text or not text.strip():
                        if self.ocr_if_needed:
                            try:
                                images = convert_from_path(
                                    str(path),
                                    dpi=OCR_DPI,
                                    first_page=i + 1,
                                    last_page=i + 1
                                )
                                if images:
                                    text = pytesseract.image_to_string(images[0], lang="mk")
                            except Exception:
                                text = ""
                    if not text or not text.strip():
                        continue

                    # Attach metadata similarly as in TXT
                    book_title = path.stem
                    author_full_name = "unknown"
                    author_bio = None
                    book_id = None
                    book_date = None
                    object_label = None

                    base_meta = {
                        "file_name": path.name,
                        "book_title": book_title,
                        "book_id": book_id,
                        "author_full_name": author_full_name,
                        "author_bio": author_bio,
                        "book_date": book_date,
                        "book_object_label": object_label,
                        "source_type": "pdf",
                        "source_path": str(path),
                        "page": i + 1,
                    }

                    # Chunk the page text
                    normalized = normalize_text(text)
                    if len(normalized) > CHUNK_SIZE:
                        subchunks = self.splitter.split_text(normalized)
                        for idx, chunk in enumerate(subchunks):
                            meta = dict(base_meta)
                            meta["chunk_index"] = idx
                            docs.append(Document(page_content=chunk, metadata=meta))
                    else:
                        meta = dict(base_meta)
                        meta["chunk_index"] = 0
                        docs.append(Document(page_content=normalized, metadata=meta))
        except Exception as e:
            print(f"[!] Error reading PDF {path}: {e}")
        return docs
processor=Preprocessor()
contents=processor.load_txt()
print(contents['393'])
print(len(contents['393'].page_content))