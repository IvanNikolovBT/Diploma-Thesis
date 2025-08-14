import os
import sys
from pathlib import Path
import re
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from typing import Dict, List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from poetry_DB import PoetryDB
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import logging
import time

from langchain.text_splitter import CharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
OCR_DPI = 300
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)
class Preprocessor:
    def __init__(self,
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP,
                 ocr_if_needed: bool = True):
        self.splitter = CharacterTextSplitter(
        chunk_size=chunk_size,       
        chunk_overlap=chunk_overlap, 
        separator="",                
        length_function=len,         
        is_separator_regex=False     
    )
        self.ocr_if_needed = ocr_if_needed
        self.db = PoetryDB()
        
        
    def _get_safe_device(self) -> str:
        """Get available device with proper error handling"""
        if not torch.cuda.is_available():
            return "cpu"
        
        try:
            
            torch.zeros(1).to("cuda")
            return "cuda"
        except RuntimeError as e:
            print(f"CUDA available but not usable: {e}")
            return "cpu"

    def _add_chunk_sequence_meta(self, chunks: List[Document]) -> List[Document]:
        for i, chunk in enumerate(chunks):
            
            book_id = chunk.metadata.get('book_id', 'unknown')
            
            chunk.metadata.update({
                "chunk_seq": i + 1,
                "total_chunks": len(chunks),
                "prev_chunk_id": f"{book_id}_chunk{i}" if i > 0 else f"{book_id}_chunk0",
                "next_chunk_id": f"{book_id}_chunk{i+2}" if i < len(chunks)-1 else f"{book_id}_chunk0",
            })
        return chunks

    def load_txt(self, directory_path: str = "extracted_books_from_images") -> List[Document]:
        all_chunks = []
        directory = Path(directory_path)
        pattern = re.compile(r'^(\d+)\.txt$')

        for file_path in directory.glob('*.txt'):
            try:
                text = file_path.read_text(encoding='utf-8')
                m = pattern.match(file_path.name)
                if not m:
                    print(f"Skipping '{file_path.name}': Filename mismatch")
                    continue

                book_id = m.group(1)
                book_info = self.db.get_book_info_by_id_in_link(book_id)

                base_metadata = {
                    "file_name": file_path.name,
                    "book_id": book_info['book_id'],
                    "book_title": book_info['book_title'],
                    "author_full_name": book_info['full_name'],
                    "author_biography": book_info['author_biography'],
                    "date": book_info['date'],
                    "object_label": book_info['object_label'],
                    "source_type": "text"
                }

    
                chunks = self.splitter.create_documents(
                    texts=[text],
                    metadatas=[base_metadata]
                )

                chunks = self._add_chunk_sequence_meta(chunks)

                all_chunks.extend(chunks)

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

        print(f"Loaded {len(all_chunks)} chunks from {len(set(doc.metadata['book_id'] for doc in all_chunks))} books")
        return all_chunks
    def load_pdf(self, path: Path) -> List[Document]:
        docs = []
        try:
            book_id = re.search(r'^(\d+)', path.stem).group(1)
            book_metadata = self.db.get_book_info_by_id_in_link(book_id)
            pdf_metadata = self._get_pdf_metadata(path)
            
            full_text = ""
            page_metadata_map = {} 
            
            with pdfplumber.open(path) as pdf:
                #print(f"\nPDF Metadata for {path.name}:")
                #print("="*50)
                #print(f"PDF Info: {pdf.metadata}")
                #print(f"Number of pages: {len(pdf.pages)}")
                #print("="*50)
                
                for i, page in enumerate(pdf.pages):
                    base_meta = {
                        **book_metadata,
                        **pdf_metadata,
                        "file_name": path.name,
                        "source_type": "pdf",
                        "source_path": str(path),
                        "page": i+1,
                        "pdf_page_size": f"{page.width}x{page.height}",
                        "is_ocr": False
                    }

                    text = page.extract_text() or ""
                    
                    if not text.strip() and self.ocr_if_needed:
                        images = convert_from_path(
                            str(path),
                            dpi=OCR_DPI,
                            first_page=i+1,
                            last_page=i+1
                        )
                        if images:
                            text = pytesseract.image_to_string(images[0], lang="mkd+eng")
                            base_meta["is_ocr"] = True
                    
                    if text.strip():
                        
                        page_separator = f"\n\n[PDF_PAGE:{i+1}]\n\n"
                        full_text += page_separator + text
                        page_metadata_map[i+1] = base_meta

            if not full_text.strip():
                return []
            
            chunks = self.splitter.split_text(full_text)
            
        
            for chunk in chunks:
                included_pages = set()
                for page_num in page_metadata_map.keys():
                    if f"[PDF_PAGE:{page_num}]" in chunk:
                        included_pages.add(page_num)
                
                if not included_pages:
                    continue
                    
                merged_meta = {
                    "included_pages": ",".join(map(str, sorted(included_pages))),  # Fixed
                    "first_page": min(included_pages),
                    "last_page": max(included_pages),
                    **page_metadata_map[next(iter(included_pages))]
                }
                
                clean_text = re.sub(r'\n?\[PDF_PAGE:\d+\]\n?', '', chunk)
                docs.append(Document(
                    page_content=clean_text,
                    metadata=merged_meta
                ))
                
        except Exception as e:
            print(f"PDF processing failed for {path}: {e}")
        
        return docs

    def load_all_pdfs(self, directory_path: str = "pdfovi/MIladinovci") -> List[Document]:
        all_documents = []
        directory = Path(directory_path)
        
        for file_path in directory.glob('*.pdf'):
            try:

                documents = self.load_pdf(file_path)
                
                if documents:  
                    all_documents.extend(documents)
                    logger.info(f"Processed {file_path.name}: {len(documents)} chunks")
                else:
                    logger.warning(f"No valid content extracted from {file_path.name}")
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {str(e)}")
                continue
                
        logger.info(f"Loaded {len(all_documents)} total chunks from {sum(1 for _ in directory.glob('*.pdf'))} PDFs")
        return all_documents
    def _get_pdf_metadata(self, path: Path) -> Dict:
        with pdfplumber.open(path) as pdf:
            return {
                "pdf_title": pdf.metadata.get("Title", ""),
                "pdf_page_count": len(pdf.pages)
            }
    

"""processor = Preprocessor()

start=time.time()
contents = processor.load_txt()
print(f'Duration {time.time()-start} seconds for all semantic')
   
book_393_chunks = contents['1']
print(f"Book 1has {len(book_393_chunks)} chunks")
for i, chunk in enumerate(book_393_chunks[:30]):  
    print(f"\nChunk {i+1}/{len(book_393_chunks)}")
    print(f"Sequence: {chunk.metadata['chunk_seq']} of {chunk.metadata['total_chunks']}")
    print(f"Prev: {chunk.metadata['prev_chunk_id']}")
    print(f"Next: {chunk.metadata['next_chunk_id']}")
    print('\n\n')
    
    print(chunk.page_content)"""
"""pdfs=processor.load_all_pdfs()  
for pdf in pdfs[:3]:
    print(pdf)
    """
"""pdf_chunks = processor.load_pdf(Path("pdfovi/MIladinovci/9.pdf"))


print(pdf_chunks)
print("\nAvailable metadata fields:")
print(pdf_chunks[0].metadata.keys())

print("\nPDF technical metadata:")
print(f"Page size: {pdf_chunks[0].metadata['pdf_page_size']}")"""