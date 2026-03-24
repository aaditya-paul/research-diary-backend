import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import re
from typing import List, Dict, Any, Optional
import html2text
import os


class EmbeddingService:
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"
        self.model = None
        self.chroma_client = None
        self.collection = None
        self._initialize()

    def _initialize(self):
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        os.makedirs(model_path, exist_ok=True)

        self.model = SentenceTransformer(self.model_name)

        chroma_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_data")
        os.makedirs(chroma_path, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="diary_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

    def _html_to_text(self, html_content: str) -> str:
        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = True
        text = converter.handle(html_content)
        return text.strip()

    def _chunk_text(self, text: str, max_length: int = 500, overlap: int = 50) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                overlap_text = " ".join(current_chunk.split()[-overlap//10:])
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text[:max_length]]

    def generate_embeddings(self, entry_id: int, html_content: str, entry_type: str, date: str):
        text_content = self._html_to_text(html_content)
        chunks = self._chunk_text(text_content)

        existing_ids = self.collection.get()["ids"]
        entry_ids_to_delete = [id for id in existing_ids if id.startswith(f"entry_{entry_id}_")]
        if entry_ids_to_delete:
            self.collection.delete(ids=entry_ids_to_delete)

        embeddings = self.model.encode(chunks)
        ids = []
        documents = []
        metadatas = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = f"entry_{entry_id}_chunk_{i}"
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append({
                "entry_id": entry_id,
                "entry_type": entry_type,
                "chunk_index": i,
                "text": chunk,
                "date": date
            })

        if ids:
            self.collection.add(
                embeddings=embeddings.tolist(),
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

    def semantic_search(self, query: str, entry_ids: List[int], n_results: int = 5) -> List[Dict[str, Any]]:
        if not entry_ids:
            return []

        query_embedding = self.model.encode([query])

        all_results = self.collection.get()

        filtered_ids = []
        filtered_metadatas = []
        filtered_documents = []

        for i, metadata in enumerate(all_results["metadatas"]):
            if metadata["entry_id"] in entry_ids:
                filtered_ids.append(all_results["ids"][i])
                filtered_metadatas.append(metadata)
                filtered_documents.append(all_results["documents"][i])

        if not filtered_ids:
            return []

        try:
            query_result = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=min(n_results, len(filtered_ids)),
                where={"entry_id": {"$in": entry_ids}}
            )

            results = []
            for i in range(len(query_result["ids"][0])):
                results.append({
                    "text": query_result["documents"][0][i],
                    "entry_id": query_result["metadatas"][0][i]["entry_id"],
                    "distance": float(query_result["distances"][0][i]) if "distances" in query_result else 0.0
                })

            return results
        except Exception as e:
            return []

    def get_all_entry_embeddings(self, entry_ids: List[int]) -> List[Dict[str, Any]]:
        if not entry_ids:
            return []

        all_results = self.collection.get()
        results = []

        for i, metadata in enumerate(all_results["metadatas"]):
            if metadata["entry_id"] in entry_ids:
                results.append({
                    "text": all_results["documents"][i],
                    "entry_id": metadata["entry_id"],
                    "chunk_index": metadata["chunk_index"],
                    "date": metadata.get("date", "")
                })

        return results


embedding_service = EmbeddingService()
