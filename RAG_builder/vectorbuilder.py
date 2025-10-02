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
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)
BM25_STORE = Path("vector_db/bm25_texts.json")
class VectorDBBuilder:
    def __init__(self,CHUNK_SIZE = 1000,CHUNK_OVERLAP = 100,model_name='sentence-transformers/all-MiniLM-L6-v2',BATCH_SIZE=3000):

        try:
            self.BATCH_SIZE=BATCH_SIZE
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device.upper()}")
            
            self.model = SentenceTransformer(
            model_name,
            device='cpu'
        )
            
            self.client = chromadb.PersistentClient(
                path="vector_db",
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
            self.preprocessor=Preprocessor(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP)
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def _clean_and_validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
    
                #logger.warning(f"Found None value in metadata field: {key} for book with book id {metadata['book_id']} and file name {metadata['file_name']}")
                if key.endswith("_id") or key in ["page", "chunk_seq"]:
                    cleaned[key] = 0
                elif key in ["is_ocr", "is_processed"]:
                    cleaned[key] = False
                else:
                    cleaned[key] = "None found"
            else:
                cleaned[key] = value
        return cleaned

    def create_vector_db(self,choice, 
                    documents: Union[Dict[str, List[Document]], List[Document]], 
                    collection_name: str = "macedonian_poetry") -> Optional[chromadb.Collection]:
        """Create or update vector database with interactive options"""
        try:
           
            existing_collections = [col.name for col in self.client.list_collections()]
            collection_exists = collection_name in existing_collections
            
            if collection_exists:
                
                
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
                    "model": "all-MiniLM-L6-v2"
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
    def query_database_semantic(
        self,
        query_text: str,
        collection_name: str = "macedonian_poetry",
        n_results: int = 5,
        filters: Optional[Dict] = None
    ) -> Dict:
       
        try:
            collection = self.client.get_collection(collection_name)
            

            query_embedding = self.model.encode(query_text).tolist()
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )
            
            return {
                "documents": results["documents"][0],
                "metadatas": results["metadatas"][0],
                "distances": results["distances"][0]
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {}
    
    def build_database_fully(self):
        """choice = input("Would you like to: \n"
                            "1. Add these documents to existing collection\n"
                            "2. Recreate the collection from scratch\n"
                            "3. Cancel operation\n"
                            "Enter choice (1-3): ").strip()"""
        choice=2
        text_docs=self.preprocessor.load_txt()
        pdf_docs=self.preprocessor.load_all_pdfs("/home/ivan/Desktop/Diplomska/pdfovi/MIladinovci")
        self.create_vector_db(choice,text_docs+pdf_docs)
        
    def build_dictionary_vdb(self):
        """Build the vector database in batches."""
        all_entries = self.preprocessor._get_all_from_o_tolkoven()  

        
        collection = self.client.get_or_create_collection(
            name="macedonian_dictionary_v2", 
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name='sentence-transformers/all-MiniLM-L6-v2')
        )
        
        for i in tqdm(range(0, len(all_entries), self.BATCH_SIZE)):
            batch_entries = all_entries[i:i + self.BATCH_SIZE]

            
            ids = [str(e['id']) for e in batch_entries]
            metadatas = [{"title": e['title'], "pos_tags": e['pos_tags'], "full_text": e['full_text']} for e in batch_entries]
            documents = [e['full_text'] for e in batch_entries]

            collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents
            )

        logger.info(f"Finished adding {len(all_entries)} entries to the vector DB.")

    def query_dictionary_semantic(
        self,
        query_text: str,
        collection_name: str = "macedonian_dictionary_v2",
        n_results: int = 1
    ) -> Dict:
        """
        Run semantic (embedding) search only.
        """
        collection = self.client.get_collection(collection_name)
        query_embedding = self.model.encode(query_text).tolist()
        sem_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        return {
            "documents": sem_results["documents"][0],
            "metadatas": sem_results["metadatas"][0],
            "distances": sem_results["distances"][0],
            "source": "semantic"
        }


    def query_dictionary_lexical(
        self,
        query_text: str,
        collection_name: str = "macedonian_dictionary_v2",
        n_results: int = 1,
        flag: int = 0
    ) -> Dict:
        """
        Dictionary lookup with flexible modes:

        flag=0 → Try lexical first, if not found use semantic fallback.  
        flag=1 → Only lexical.  
        flag=2 → Only semantic.  
        flag=3 → Return both lexical and semantic.  
        """
        try:
            collection = self.client.get_collection(collection_name)

            
            lexical_results = collection.get(
                where={"title": query_text},
                include=["documents", "metadatas"]
            )

            if flag == 0:
                if lexical_results.get("documents"):
                    return {
                        "documents": lexical_results["documents"],
                        "metadatas": lexical_results["metadatas"],
                        "source": "lexical"
                    }
                else:
                    print(f"Didn’t find an exact match for '{query_text}', searching semantically…")
                    return self.query_dictionary_semantic(query_text, collection_name, n_results)

            elif flag == 1:
                if lexical_results.get("documents"):
                    return {
                        "documents": lexical_results["documents"],
                        "metadatas": lexical_results["metadatas"],
                        "source": "lexical"
                    }
                else:
                    return {"documents": [], "metadatas": [], "source": "lexical"}

            elif flag == 2:
                return self.query_dictionary_semantic(query_text, collection_name, n_results)

            elif flag == 3:
                semantic_result = self.query_dictionary_semantic(query_text, collection_name, n_results)
                return {
                    "lexical": {
                        "documents": lexical_results.get("documents", []),
                        "metadatas": lexical_results.get("metadatas", []),
                        "source": "lexical"
                    },
                    "semantic": semantic_result
                }

            else:
                raise ValueError("Invalid flag value. Must be 0, 1, 2, or 3.")

        except Exception as e:
            logger.error(f"Dictionary query failed: {e}")
            return {}


"""if __name__ == "__main__":
    try:
 
        logger.info("Initializing vector database builder...")
        builder = VectorDBBuilder()
        builder.build_database_fully()
        choice=2
        logger.info("Initializing document preprocessor...")
        processor = Preprocessor()

        logger.info("Loading text documents...")
        choice = int(input("Would you like to: \n"
                            "1.Build?\n2.Query").strip())
        if choice==1:
            start_time=time.time()
            collection=builder.build_database_fully()
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
                                logger.info(f"Text: {text}")
                    else:
                        logger.warning("No results returned from query")
                except Exception as query_error:
                        logger.error(f"Query failed: {query_error}")
                
        elif choice==2:
            results=builder.query_database_semantic('Петре М. Андреевски песни за љубов')
            suma=0
            for i  in range(len(results)):
                print(results["documents"][i])
                print(results["metadatas"][i])
                print(results["distances"][i])
                suma+=float(results['distances'][i])
                
            suma=suma/(len(results))
            print(f'average len is {suma}')
    except Exception as main_error:
        logger.error(f"Main execution failed: {main_error}")"""
        
#'sentence-transformers/all-MiniLM-L6-v2'
  #0.4695906440416972  CHUNK_SIZE = 1400 CHUNK_OVERLAP = 300     '
  #0.45668325821558636 CHUNK_SIZE = 1500 CHUNK_OVERLAP = 200 
  #0.45740966002146405 CHUNK_SIZE = 900 CHUNK_OVERLAP = 200
  #0.4370685617129008 CHUNK_SIZE = 1200CHUNK_OVERLAP = 200
  # 0.4704437851905823 CHUNK_SIZE = 1000 CHUNK_OVERLAP = 100