import torch
from pathlib import Path
from typing import Union, Dict, List, Optional, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from preprocessor import Preprocessor
import logging
import time 

logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)

class VectorDBBuilder:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):

        try:
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device.upper()}")
            
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": self.device},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 512 if self.device == "cuda" else 128
                }
            )
            
            self.client = chromadb.PersistentClient(
                path="vector_db",
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
            self.preprocessor=Preprocessor()
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def _clean_and_validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
    
                logger.warning(f"Found None value in metadata field: {key} for book with book id {metadata['book_id']} and file name {metadata['file_name']}")
                if key.endswith("_id") or key in ["page", "chunk_seq"]:
                    cleaned[key] = 0
                elif key in ["is_ocr", "is_processed"]:
                    cleaned[key] = False
                else:
                    cleaned[key] = "None found"
            else:
                cleaned[key] = value
        return cleaned

    def create_vector_db(self, 
                    documents: Union[Dict[str, List[Document]], List[Document]], 
                    collection_name: str = "macedonian_poetry") -> Optional[chromadb.Collection]:
        """Create or update vector database with interactive options"""
        try:
           
            existing_collections = [col.name for col in self.client.list_collections()]
            collection_exists = collection_name in existing_collections
            
            if collection_exists:
                print(f"\nVector database '{collection_name}' already exists.")
                choice = input("Would you like to: \n"
                            "1. Add these documents to existing collection\n"
                            "2. Recreate the collection from scratch\n"
                            "3. Cancel operation\n"
                            "Enter choice (1-3): ").strip()
                
                if choice == "3":
                    print("Operation cancelled.")
                    return self.client.get_collection(collection_name)
                elif choice == "2":
                    print(f"Deleting existing collection '{collection_name}'...")
                    self.client.delete_collection(collection_name)
                    collection_exists = False

            
            if not documents:
                logger.error("No documents provided")
                return None

            
            all_chunks = []
            if isinstance(documents, dict):
                for book_id, chunks in documents.items():
                    if not chunks:
                        logger.warning(f"No chunks found for book {book_id}")
                        continue
                    all_chunks.extend(chunks)
            else:
                all_chunks = documents

            if not all_chunks:
                logger.error("No valid chunks found in documents")
                return None

           
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "device": self.device,
                    "model": self.embedding_model.model_name
                }
            )

            
            batch_size = 512 if self.device == "cuda" else 128
            total_chunks = len(all_chunks)
            action = "Updating" if collection_exists else "Creating"
            logger.info(f"{action} vector database with {total_chunks} chunks in batches of {batch_size}")

            processed_chunks = 0
            start_time = time.time()
            
            for i in range(0, total_chunks, batch_size):
                batch = all_chunks[i:i + batch_size]
                try:

                    documents_batch = []
                    metadatas_batch = []
                    ids_batch = []
                    
                    for j, chunk in enumerate(batch):
                        if not isinstance(chunk, Document):
                            continue
                        if not hasattr(chunk, 'page_content') or not chunk.page_content:
                            continue
                        
                        cleaned_metadata = self._clean_and_validate_metadata(chunk.metadata)
                        documents_batch.append(chunk.page_content)
                        metadatas_batch.append(cleaned_metadata)
                        ids_batch.append(f"chunk_{i+j}")

                    if documents_batch:

                        collection.upsert(
                            documents=documents_batch,
                            metadatas=metadatas_batch,
                            ids=ids_batch
                        )
                        processed_chunks += len(documents_batch)

                    if i % (batch_size * 10) == 0:
                        elapsed = time.time() - start_time
                        print(f"Processed {processed_chunks}/{total_chunks} chunks "
                            f"({elapsed:.1f}s elapsed)")

                except Exception as batch_error:
                    logger.error(f"Batch {i//batch_size} failed: {batch_error}")
                    continue

            logger.info(f"Completed processing {total_chunks} chunks in {time.time()-start_time:.1f}s")
            print(f"\nVector database '{collection_name}' ready with {collection.count()} chunks")
            return collection

        except Exception as e:
            logger.error(f"Vector DB operation failed: {e}")
            return None

if __name__ == "__main__":
    try:
 
        logger.info("Initializing vector database builder...")
        builder = VectorDBBuilder()
        
        logger.info("Initializing document preprocessor...")
        processor = Preprocessor()

        logger.info("Loading text documents...")
        txt_contents = {}
        try:
            txt_contents = processor.load_txt()
            if not txt_contents:
                logger.error("No text documents loaded")
            else:
                sample_book_id = next(iter(txt_contents))
                sample_chunk = txt_contents[sample_book_id][0]
                logger.debug(f"Sample metadata for book {sample_book_id}:")
                logger.debug(sample_chunk.metadata)
        except Exception as load_error:
            logger.error(f"Document loading failed: {load_error}")

        if txt_contents:
            print("Creating vector database...")
            start_time=time.time()
            collection = builder.create_vector_db(txt_contents)
            end_time=time.time()
            if collection:
                logger.info(f'Created collection: took {end_time-start_time} duration')
                try:
                    logger.info("Running test query...")
                    results = collection.query(
                        query_texts=["Петре м андреевски"],
                        n_results=20,
                        where={"author_full_name": {"$ne": ""}}  
                    )
                    
                    if results and 'documents' in results:
                        logger.info("\nQuery results:")
                        for i, (text, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                            logger.info(f"\nResult {i+1}:")
                            logger.info(f"Author: {meta.get('author_full_name', 'Unknown')}")
                            logger.info(f"Book: {meta.get('book_title', 'Unknown')}")
                            logger.info(f"Text: {text}...")
                    else:
                        logger.warning("No results returned from query")
                except Exception as query_error:
                    logger.error(f"Query failed: {query_error}")
        else:
            logger.error("No documents available for vector DB creation")

    except Exception as main_error:
        logger.error(f"Main execution failed: {main_error}")