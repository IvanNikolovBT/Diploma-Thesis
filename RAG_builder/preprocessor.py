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
from langchain_community.document_loaders import PDFMinerLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from poetry_DB import PoetryDB

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
OCR_DPI = 300

class Preprocessor:
    def __init__(self,
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP,
                 ocr_if_needed: bool = True):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]  
        )
        self.ocr_if_needed = ocr_if_needed
        self.db = PoetryDB()

    def _add_chunk_sequence_meta(self, chunks: List[Document]) -> List[Document]:
        
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_seq": i + 1,
                "total_chunks": len(chunks),
                "prev_chunk_id": f"{chunk.metadata['book_id']}_chunk{i}" if i > 0 else None,
                "next_chunk_id": f"{chunk.metadata['book_id']}_chunk{i+2}" if i < len(chunks)-1 else None
            })
        return chunks

    def load_txt(self, directory_path: str = "extracted_books_from_images") -> Dict[str, List[Document]]:
        chunked_documents = {}
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
                chunked_documents[book_id] = chunks

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

        print(f"Loaded {sum(len(v) for v in chunked_documents.values())} chunks from {len(chunked_documents)} books")
        return chunked_documents

    def load_pdf(self, path: Path) -> List[Document]:
        docs = []
        try:
            book_id = re.search(r'^(\d+)', path.stem).group(1)
            book_metadata = self.db.get_book_info_by_id_in_link(book_id)
            
            pdf_metadata = self._get_pdf_metadata(path)
            
            with pdfplumber.open(path) as pdf:
                print(f"\nPDF Metadata for {path.name}:")
                print("="*50)
                print(f"PDF Info: {pdf.metadata}")
                print(f"Number of pages: {len(pdf.pages)}")
                print("="*50)
                
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
                    
                    if not text.strip():
                        continue


                    page_chunks = self.splitter.create_documents(
                        texts=[text],
                        metadatas=[base_meta]
                    )
                    docs.extend(self._add_chunk_sequence_meta(page_chunks))

        except Exception as e:
            print(f"PDF processing failed for {path}: {e}")
        
        return docs

    def _get_pdf_metadata(self, path: Path) -> Dict:
        """Extract PDF-specific metadata"""
        with pdfplumber.open(path) as pdf:
            return {
                "pdf_title": pdf.metadata.get("Title", ""),
                "pdf_author": pdf.metadata.get("Author", ""),
                "pdf_creator": pdf.metadata.get("Creator", ""),
                "pdf_producer": pdf.metadata.get("Producer", ""),
                "pdf_creation_date": pdf.metadata.get("CreationDate", ""),
                "pdf_mod_date": pdf.metadata.get("ModDate", ""),
                "pdf_version": pdf.metadata.get("PDFVersion", ""),
                "pdf_page_count": len(pdf.pages)
            }

processor = Preprocessor()
contents = processor.load_txt()


book_393_chunks = contents['393']
print(f"Book 393 has {len(book_393_chunks)} chunks")
for i, chunk in enumerate(book_393_chunks[:3]):  
    print(f"\nChunk {i+1}/{len(book_393_chunks)}")
    print(f"Sequence: {chunk.metadata['chunk_seq']} of {chunk.metadata['total_chunks']}")
    print(f"Prev: {chunk.metadata['prev_chunk_id']}")
    print(f"Next: {chunk.metadata['next_chunk_id']}")
    print(chunk.page_content[:100] + "...")
    
    
pdf_chunks = processor.load_pdf(Path("pdfovi/MIladinovci/9.pdf"))


print(pdf_chunks)
print("\nAvailable metadata fields:")
print(pdf_chunks[0].metadata.keys())

print("\nPDF technical metadata:")
print(f"Creator: {pdf_chunks[0].metadata['pdf_creator']}")
print(f"Page size: {pdf_chunks[0].metadata['pdf_page_size']}")