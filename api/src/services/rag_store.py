"""
create a response for chatbot using RAG with Qdrant
"""

import logging
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams
from src.core.config import settings
from src.core.models import Source

logger = logging.getLogger(__name__)


class RAGService:
    """
    Service for RAG
    """

    def __init__(self) -> None:
        self.client = None
        self.embeddings = None
        self.text_splitter = None
        self._initialize()

    def _initialize(self):
        try:
            # init qdrant db client
            self.client = QdrantClient(url=settings.QDRANT_URL)

            # init embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            # init splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len,
            )

            # create collection
            self._create_collection_if_not_exist()

            logger.info("RAG service init successfuly")

        except Exception as e:
            logger.error(f"Error init RAG service : {e}")
            raise

    def _create_collection_if_not_exist(self):
        """
        create QdrantDB collection if it does't exist
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if settings.QDRANT_COLLECTION not in collection_names:
                self.client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION,
                    vectors_config=VectorParams(
                        size=settings.QDRANT_VECTOR_SIZE, distance=Distance.COSINE
                    ),
                )
                logger.info(f"create collection: {settings.QDRANT_COLLECTION}")

        except Exception as e:
            logger.error(f"error creating collection: {e}")
            raise

    def _extract_text_from_file(self, content: bytes, filename: str) -> str:
        """
        Extract text from reference.
        """
        ext = filename.lower().split(".")[-1] if "." in filename else ""

        if ext == "pdf":
            reader = PdfReader(BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)

        if ext in ("txt", "md", "text"):
            return content.decode("utf-8", errors="replace")

        raise ValueError(f"Formato no soportado: {ext}. Use .pdf o .txt")

    def add_documents(
        self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        add documents to vector store

        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document

        Returns:
            Number of chunks added
        """
        try:
            all_chunks = []
            all_metadatas = []

            for idx, doc in enumerate(documents):
                # split document into chunks
                chunks = self.text_splitter.split_text(doc)

                # create metadata for each chunk
                metadata = metadatas[idx] if metadatas and idx < len(metadatas) else {}
                chunk_metadatas = [
                    {**metadata, "chunk_index": i, "source_doc_index": idx}
                    for i in range(len(chunks))
                ]

                all_chunks.extend(chunks)
                all_metadatas.extend(chunk_metadatas)

            # create embeddings
            embeddings = self.embeddings.embed_documents(all_chunks)

            # create points for QdrantDB
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"text": chunk, **metadata},
                )
                for chunk, embedding, metadata in zip(
                    all_chunks, embeddings, all_metadatas
                )
            ]

            # Upload to QdrantDB
            self.client.upsert(
                collection_name=settings.QDRANT_COLLECTION, points=points
            )

            logger.info(f"Added {len(points)} chunks to vector store")
            return len(points)

        except Exception as e:
            logger.error(f"Error for add document to RAG store: {e}")
            raise

    def add_from_file(
        self,
        content: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        text = self._extract_text_from_file(content, filename)
        meta = metadata or {}
        meta.setdefault("source", filename)
        return self.add_documents(documents=[text], metadatas=[meta])

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Source]:
        """
        search for relevant documents

        Args:
            query: search query
            top_k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of Source objects with relevant documents
        """

        try:
            if top_k is None:
                top_k = settings.TOP_K_RESULTS

            # generate query embeddings
            query_embedding = self.embeddings.embed_query(query)

            # build filter if provided
            search_filter = None
            if filter_dict:
                search_filter = Filter(**filter_dict)

            # search with query points in QdrantDB
            response = self.client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=query_embedding,
                limit=top_k,
                query_filter=search_filter,
            )

            search_results = response.points

            # convert to source objects
            sources = []
            for result in search_results:
                source = Source(
                    content=result.payload.get("text", ""),
                    source=result.payload.get("source", "Unknown"),
                    relevance_score=result.score,
                )
                sources.append(source)

            logger.info(f"found {len(sources)} relevant documents for query")
            return sources

        except Exception as e:
            logger.error(f"error search documents: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        get statistics about the vector store
        """

        try:
            collection_info = self.client.get_collection(
                collection_name=settings.QDRANT_COLLECTION
            )
            return {
                "total_documents": collection_info.points_count,
                "collection_name": settings.QDRANT_COLLECTION,
                "vector_size": settings.QDRANT_VECTOR_SIZE,
            }
        except Exception as e:
            logger.error(f"error getting stats: {e}")
            return {}

    def delete_collection(self):
        """
        delete the entire collection (warning: use with caution)
        """

        try:
            self.client.delete_collection(collection_name=settings.QDRANT_COLLECTION)
            logger.info(f"deleted collection: {settings.QDRANT_COLLECTION}")
            # recreate empty collection
            self._create_collection_if_not_exist()
        except Exception as e:
            logger.error(f"error deleting collections: {e}")
            raise


# singlenton instance
_rag_service = None


def get_rag_service() -> RAGService:
    """
    Get or create RAG service singleton
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
