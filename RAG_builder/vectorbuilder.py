import torch
from pathlib import Path
from typing import Union, Dict, List, Optional, Any
from langchain_core.documents import Document
import chromadb
from RAG_builder.preprocessor import Preprocessor
import logging
import time 
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModel
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
    
                if key.endswith("_id") or key in ["page", "chunk_seq"]:
                    cleaned[key] = 0
                elif key in ["is_ocr", "is_processed"]:
                    cleaned[key] = False
                else:
                    cleaned[key] = "None found"
            else:
                cleaned[key] = value
        return cleaned
    
    def get_macedonizer_embeddings(self,
        texts: List[str],
        tokenizer,
        model,
        device,
        batch_size: int = 64,
        max_len: int = 512
    ) -> np.ndarray:
        all_emb = []
        model.eval()
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=max_len,
                padding='max_length',
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            masked = hidden * mask
            summed = masked.sum(1)
            lengths = mask.sum(1).clamp(min=1.0)
            mean_pooled = summed / lengths
            all_emb.append(mean_pooled.cpu().numpy())
        embeddings = np.vstack(all_emb)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        return embeddings

    def create_vector_db_macedonizer(
        self,
        choice: str,
        documents: Union[Dict[str, List[Document]], List[Document]],
        collection_name: str = "macedonian_poetry",
        batch_size: int = 64,
        embedding_batch_size: int = 64,
        max_len: int = 512,
    ) -> Optional[chromadb.Collection]:
        existing_collections = [c.name for c in self.client.list_collections()]
        collection_exists = collection_name in existing_collections
        if collection_exists:
            if choice == "3":
                print("Operation cancelled.")
                return self.client.get_collection(collection_name)
            elif choice == "2":
                print(f"Deleting existing collection '{collection_name}'...")
                self.client.delete_collection(collection_name)
                collection_exists = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading macedonizer/mk-roberta-base on {device}...")
        tokenizer = AutoTokenizer.from_pretrained("macedonizer/mk-roberta-base")
        model = AutoModel.from_pretrained("macedonizer/mk-roberta-base")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.to(device)
        model.eval()
        if not documents:
            logger.error("No documents provided")
            return None
        all_chunks: List[Document] = []
        if isinstance(documents, dict):
            for book_id, chunks in documents.items():
                if not chunks:
                    logger.warning(f"No chunks for book {book_id}")
                    continue
                all_chunks.extend(chunks)
        else:
            all_chunks = documents
        if not all_chunks:
            logger.error("No valid chunks found")
            return None
        total_chunks = len(all_chunks)
        print(f"Preparing to embed {total_chunks} chunks...")
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        processed = 0
        start_time = time.time()
        for i in range(0, total_chunks, batch_size):
            batch_docs = all_chunks[i:i + batch_size]
            texts = []
            metadatas = []
            ids = []
            for j, doc in enumerate(batch_docs):
                if not isinstance(doc, Document) or not getattr(doc, "page_content", None):
                    continue
                cleaned_meta = self._clean_and_validate_metadata(doc.metadata)
                texts.append(doc.page_content)
                metadatas.append(cleaned_meta)
                ids.append(f"chunk_{i + j}")
            if not texts:
                continue
            embeddings = self.get_macedonizer_embeddings(
                texts=texts,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=embedding_batch_size,
                max_len=max_len
            )
            collection.upsert(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids
            )
            processed += len(texts)
            if (i // batch_size) % 5 == 0:
                elapsed = time.time() - start_time
                print(f"  -> {processed}/{total_chunks} chunks ({elapsed:.1f}s)")
        total_time = time.time() - start_time
        print(f"\nVector DB '{collection_name}' ready with {collection.count()} chunks in {total_time:.1f}s")
        return collection
    def legacy_create_vector_db(self,choice, 
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
        collection_name: str = "makedonizer_poetry_db",
        n_results: int = 5,
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Query poetry DB using Macedonizer embeddings (384 dim).
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            tokenizer = AutoTokenizer.from_pretrained("macedonizer/mk-roberta-base")
            model = AutoModel.from_pretrained("macedonizer/mk-roberta-base")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.to(device)
            model.eval()

            inputs = tokenizer(
                [query_text.lower()],
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                hidden = outputs.last_hidden_state

            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            embedding = torch.nn.functional.normalize(pooled, p=2, dim=1)
            query_embedding = embedding[0].cpu().numpy().tolist()

            collection = self.client.get_collection(collection_name)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )

            return {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else []
            }

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"documents": [], "metadatas": [], "distances": []}
    def legacy_query_database_semantic(
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
    
    def build_database_fully(self, choice=2, database='makedonizer_poetry_db'):
        text_docs = self.preprocessor.load_txt()
        pdf_docs = self.preprocessor.load_all_pdfs("/home/ivan/Desktop/Diplomska/pdfovi/MIladinovci")
        
        all_docs = text_docs + pdf_docs
        
        for doc in all_docs:
            if hasattr(doc, 'page_content') and isinstance(doc.page_content, str):
                doc.page_content = doc.page_content.lower()
        
        self.create_vector_db_macedonizer(choice, all_docs, collection_name=database)
        
    def build_dictionary_vdb_macedonizer(self):
        all_entries = self.preprocessor._get_all_from_o_tolkoven()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("macedonizer/mk-roberta-base")
        model = AutoModel.from_pretrained("macedonizer/mk-roberta-base")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.to(device)
        model.eval()

        collection = self.client.get_or_create_collection(
            name="macedonian_dictionary_macedonizer",
            metadata={"hnsw:space": "cosine"}
        )

        for i in tqdm(range(0, len(all_entries), self.BATCH_SIZE)):
            batch = all_entries[i:i + self.BATCH_SIZE]
            
            texts = [e['full_text'].lower() for e in batch]
            ids = [str(e['id']) for e in batch]
            metadatas = [{"title": e['title'], "pos_tags": e['pos_tags']} for e in batch]

            embeddings = self.get_macedonizer_embeddings(
                texts=texts,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=32,
                max_len=512
            )

            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )

        logger.info(f"Finished adding {len(all_entries)} entries to Macedonizer dictionary DB.")
    def legacy_build_dictionary_vdb(self):
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
        collection_name: str = "macedonian_dictionary_macedonizer",
        n_results: int = 1
    ) -> Dict:
 
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            tokenizer = AutoTokenizer.from_pretrained("macedonizer/mk-roberta-base")
            model = AutoModel.from_pretrained("macedonizer/mk-roberta-base")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.to(device)
            model.eval()

            inputs = tokenizer(
                [query_text.lower()],
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                hidden = outputs.last_hidden_state

            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            embedding = torch.nn.functional.normalize(pooled, p=2, dim=1)
            query_embedding = embedding[0].cpu().numpy().tolist()

            collection = self.client.get_collection(collection_name)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

            return {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "source": "semantic_macedonizer"
            }

        except Exception as e:
            logger.error(f"Macedonizer dictionary query failed: {e}")
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "source": "semantic_macedonizer"
            }
    def legacy_query_dictionary_semantic(
        self,
        query_text: str,
        collection_name: str = "macedonian_dictionary_v2",
        n_results: int = 1
    ) -> Dict:

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
    def query_lemmas(self, lemmas, collection_name="macedonian_dictionary_v2", n_results=1, flag=0):
    
        
        prefix_words = [
            "почетна",
            "корпус",
            "топоними",
            "кратенки",
            "toggle sidebar"
        ]
        prefix = "\n".join(prefix_words) + "\n"

        results = []
        for lemma in lemmas:
            try:
                result = self.query_dictionary_lexical(lemma, collection_name, n_results, flag)
                text = result['documents'][0]

                
                if text.startswith(prefix):
                    text = text[len(prefix):]

                found = text.split('\n', 1)[0]
                source = result['source']

                if source == 'semantic':
                    distance = result['distances'][0]
                else:
                    distance = 0

                results.append({
                    "lemma": lemma,
                    "lemma_found": found,
                    "text": text + "\n",
                    "source": source,
                    "distance": distance
                })

            except Exception as e:
                print(f"Query failed for lemma '{lemma}': {e}")
                    
        return results
