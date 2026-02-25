from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """chat messahe model"""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    """request model for chatbot endpoint"""

    message: str = Field(..., min_length=1, description="user message")
    conversation_id: Optional[str] = Field(
        None, description="conversation id for context"
    )
    include_sources: bool = Field(
        True, description="include source documents in response"
    )
    force_web_search: bool = False


class Source(BaseModel):
    """source document metadata"""

    content: str
    source: str
    # source_url = Optional[str] = None
    relevance_score: Optional[float] = None
    # type: Literal["local"]


class ChatResponse(BaseModel):
    """response model for chatbot endpoint"""

    response: str
    conversation_id: str
    tool_used: Literal["rag", "tavily", "direct", "agent_decision", "multitool"] = (
        "agent_decision"
    )
    sources: Optional[List[Source]] = None
    thought_process: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """health check response"""

    status: str
    timestamp: datetime
    services: Dict[str, Any]


class DocumentUpload(BaseModel):
    """model structure for document upload"""

    title: str
    content: str
    category: Optional[str] = "general"
    metadata: Optional[Dict[str, Any]] = None


class EmbeddingsStats(BaseModel):
    """statistics about embeddings"""

    total_documents: int
    collection_name: str
    vector_size: int


class VectorSearchQuery(BaseModel):
    """model for make structure query to Qdrant"""

    query: str
    filter_category: Optional[str] = None
    limit: int = 5
