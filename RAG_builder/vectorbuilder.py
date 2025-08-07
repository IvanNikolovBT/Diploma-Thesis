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
# Configure logging
logging.basicConfig(level=logging.INFO)  # Changed to DEBUG for more detailed output
logger = logging.getLogger(__name__)

class VectorDBBuilder:
    def __init__(self, embedding_model: str = "intfloat/multilingual-e5-large"):
        """Initialize with proper error handling and GPU support"""
        try:
            # Device configuration
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device.upper()}")
            
            # Initialize embeddings with proper settings
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": self.device},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 128 if self.device == "cuda" else 32
                }
            )
            
            # ChromaDB configuration
            self.client = chromadb.PersistentClient(
                path="vector_db",
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def _clean_and_validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata and log any None values found"""
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
    
                logger.warning(f"Found None value in metadata field: {key}")
                # Replace None with appropriate defaults
                if key.endswith("_id") or key in ["page", "chunk_seq"]:
                    cleaned[key] = 0
                elif key in ["is_ocr", "is_processed"]:
                    cleaned[key] = False
                else:
                    cleaned[key] = "None found"
            else:
                cleaned[key] = value
        return cleaned

    def _log_problematic_metadata(self, metadata: Dict[str, Any]):
        """Log detailed information about problematic metadata"""
        problematic_fields = {k: v for k, v in metadata.items() if v is None}
        if problematic_fields:
            logger.debug(f"Problematic metadata fields: {problematic_fields}")
            logger.debug(f"Complete metadata: {metadata}")

    def create_vector_db(self, 
                       documents: Union[Dict[str, List[Document]], List[Document]], 
                       collection_name: str = "macedonian_poetry") -> Optional[chromadb.Collection]:
        """Create vector database with robust error handling"""
        try:
            # Validate input
            if not documents:
                logger.error("No documents provided")
                return None

            # Flatten documents
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

            # Create collection
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "device": self.device,
                    "model": self.embedding_model.model_name
                }
            )

            # Batch processing with progress logging
            batch_size = 512 if self.device == "cuda" else 128
            total_chunks = len(all_chunks)
            logger.info(f"Processing {total_chunks} chunks in batches of {batch_size}")

            for i in range(0, total_chunks, batch_size):
                batch = all_chunks[i:i + batch_size]
                try:
                    # Prepare batch data with cleaned metadata
                    documents_batch = []
                    metadatas_batch = []
                    ids_batch = []
                    
                    for j, chunk in enumerate(batch):
                        previous=time.time()
                        if not isinstance(chunk, Document):
                            logger.warning(f"Invalid document type: {type(chunk)}")
                            continue
                        
                        if not hasattr(chunk, 'page_content') or not chunk.page_content:
                            logger.warning("Skipping empty document")
                            continue
                        
                        # Log problematic metadata before cleaning
                        self._log_problematic_metadata(chunk.metadata)
                        
                        # Clean the metadata
                        cleaned_metadata = self._clean_and_validate_metadata(chunk.metadata)
                        
                        documents_batch.append(chunk.page_content)
                        metadatas_batch.append(cleaned_metadata)
                        ids_batch.append(f"chunk_{i+j}")

                    if not documents_batch:
                        continue

                    # Add to collection
                    collection.add(
                        documents=documents_batch,
                        metadatas=metadatas_batch,
                        ids=ids_batch
                    )
                    
                    if i % (batch_size * 10) == 0:  # Log progress every 10 batches
                        print(f"Processed {min(i + batch_size, total_chunks)}/{total_chunks} chunks")
                        current=time.time()
                        print(f"Time needed {current-previous} chunks")
                except Exception as batch_error:
                    logger.error(f"Batch {i//batch_size} failed: {batch_error}")
                    continue

            logger.info(f"Successfully created vector database with {total_chunks} chunks")
            return collection

        except Exception as e:
            logger.error(f"Vector DB creation failed: {e}")
            return None

if __name__ == "__main__":
    try:
        # Initialize components
        logger.info("Initializing vector database builder...")
        builder = VectorDBBuilder()
        
        logger.info("Initializing document preprocessor...")
        processor = Preprocessor()

        # Process documents with error handling
        logger.info("Loading text documents...")
        txt_contents = {}
        try:
            txt_contents = processor.load_txt()
            if not txt_contents:
                logger.error("No text documents loaded")
            else:
                # Print sample metadata for inspection
                sample_book_id = next(iter(txt_contents))
                sample_chunk = txt_contents[sample_book_id][0]
                logger.debug(f"Sample metadata for book {sample_book_id}:")
                logger.debug(sample_chunk.metadata)
        except Exception as load_error:
            logger.error(f"Document loading failed: {load_error}")

        # Create vector database
        if txt_contents:
            print("Creating vector database...")
            start_time=time.time()
            collection = builder.create_vector_db(txt_contents)
            end_time=time.time()
            if collection:
                # Example query with error handling
                logger.info(f'Created collection: took {end_time-start_time} duration')
                try:
                    logger.info("Running test query...")
                    results = collection.query(
                        query_texts=["Македонска народна песна"],
                        n_results=20,
                        where={"author_full_name": {"$ne": ""}}  # Filter out empty authors
                    )
                    
                    if results and 'documents' in results:
                        logger.info("\nQuery results:")
                        for i, (text, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                            logger.info(f"\nResult {i+1}:")
                            logger.info(f"Author: {meta.get('author_full_name', 'Unknown')}")
                            logger.info(f"Book: {meta.get('book_title', 'Unknown')}")
                            logger.info(f"Text: {text[:200]}...")
                    else:
                        logger.warning("No results returned from query")
                except Exception as query_error:
                    logger.error(f"Query failed: {query_error}")
        else:
            logger.error("No documents available for vector DB creation")

    except Exception as main_error:
        logger.error(f"Main execution failed: {main_error}")