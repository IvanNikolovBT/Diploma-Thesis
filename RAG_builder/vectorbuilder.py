import torch
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
from typing import Union, Dict, List

class VectorDBBuilder:
    def __init__(self, embedding_model: str = "intfloat/multilingual-e5-large"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device.upper()}")
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={
                "device": self.device,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32
            },
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 128 if self.device == "cuda" else 32
            }
        )
        
        settings = chromadb.Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        )
        
        if self.device == "cuda":
            settings.require("hnswlib") 
            
        self.client = chromadb.PersistentClient(
            path="vector_db",
            settings=settings
        )

    def create_vector_db(self, documents: Union[Dict[str, List[Document]], List[Document]], 
                        collection_name: str = "macedonian_poetry"):

        all_chunks = (
            [chunk for chunks in documents.values() for chunk in chunks] 
            if isinstance(documents, dict) 
            else documents
        )


        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "device": self.device,
                "model": self.embedding_model.model_name
            },
            embedding_function=self.embedding_model.embed_documents
        )


        batch_size = 512 if self.device == "cuda" else 128
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            
            collection.add(
                documents=[chunk.page_content for chunk in batch],
                metadatas=[chunk.metadata for chunk in batch],
                ids=[f"chunk_{i+j}" for j in range(len(batch))]
            )

        print(f"Created vector database with {len(all_chunks)} chunks (GPU: {self.device == 'cuda'})")
        return collection

    def query(self, collection_name: str, query_text: str, 
              n_results: int = 5, **metadata_filters):
        """GPU-accelerated query with metadata filtering"""
        collection = self.client.get_collection(collection_name)
        

        where = {}
        for k, v in metadata_filters.items():
            if isinstance(v, list):
                where[k] = {"$in": v}
            else:
                where[k] = {"$eq": v}
        
        return collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["metadatas", "distances"]
        )


if __name__ == "__main__":
    builder = VectorDBBuilder()
    
    processor = Preprocessor()
    txt_contents = processor.load_txt()
    
    collection = builder.create_vector_db(txt_contents)
    
    results = builder.query(
        collection_name="macedonian_poetry",
        query_text="Македонска народна песна",
        n_results=3,
        author_full_name="Miladinovci",
        date={"$gte": "1850"}  #
    )
    
    print("\nTop results:")
    for i, (meta, dist) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
        print(f"\nResult {i+1} (distance: {dist:.3f}):")
        print(f"Author: {meta['author_full_name']}")
        print(f"Book: {meta['book_title']}")
        print(f"Content: {meta['text'][:100]}...")