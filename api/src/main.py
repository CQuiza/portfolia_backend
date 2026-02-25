"""
FastAPI application - Gateway
"""

import logging
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
from src.core.config import settings
from src.core.models import (
    ChatRequest,
    ChatResponse,
    DocumentUpload,
    EmbeddingsStats,
    HealthResponse,
)
from src.routes import auth
from src.services.agent import get_agent
from src.services.rag_store import get_rag_service
from src.services.tavily_search import get_tavily_service
from src.utils.check_admin import check_admin_access

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

# Logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    # force=True,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for this application
    """
    logger.info("Starting Portfolio Chatbot API ...")

    # Init Services
    try:
        get_rag_service()
        get_tavily_service()
        get_agent()
        logger.info("services initialized successfully")
    except Exception as e:
        logger.error(f"Error init services: {e}")

    yield
    logger.info("----Shutdown portfolio Chatbot API----")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.API_VERSION,
    openapi_url="/openapi.json",
    docs_url="/docs"
    lifespan=lifespan,
    description="AI-powered portfolio chatbot with RAG and web search capabilities - Cristhian Quiza",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])


@app.get("/", tags=["general"])
async def root():
    """root endpoint"""
    return {
        "message": "PortfolIA Chatbot API",
        "version": settings.API_VERSION,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    try:
        rag_service = get_rag_service()
        tavily_service = get_tavily_service()

        services = {
            "rag": "healthy" if rag_service else "unavailable",
            "tavily": "healthy" if tavily_service.is_available() else "unavailable",
            "llm": "healthy",
        }

        logger.info("healthy successfully")

        return HealthResponse(
            status="healthy", timestamp=datetime.now(), services=services
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unhealthy")


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest):
    """
    Main chatbot endpoint to processes user messages

    The agent will automatically decide whether to:
    - Use RAG for portfolio-related questions
    - Use Tavily for general web searches
    - use both in multitool for combined questions
    - Respond directly for casual conversation
    """
    try:
        # generate conversation id
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Process query
        agent = get_agent()
        response = agent.process_query(
            query=request.message, conversation_id=conversation_id
        )

        # filter sources if not requested
        if not request.include_sources:
            response.sources = None

        return response

    except Exception as e:
        logger.error(f"Error in process the user request to chatboy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload_file", tags=["Documents"])
async def upload_file(
    file: UploadFile = File(...), is_admin: bool = Depends(check_admin_access)
):
    """
    Upload a document to the RAG knowledge base

    This endpoint allows you to add new information about yourself
    that the chatbot can reference when answering questions.

    Allowed extensions: .pdf, .txt, .md
    """
    try:
        ext = (
            "." + file.filename.rsplit(".", 1)[-1].lower()
            if "." in file.filename
            else ""
        )
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Formato no soportado. Permitidos: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Archivo vacío")

        rag_service = get_rag_service()
        metadata = {
            "title": file.filename,
            "category": "upload",
            "source": file.filename,
            "upload_date": datetime.now().isoformat(),
        }

        chunks_added = rag_service.add_from_file(
            content=content,
            filename=file.filename,
            metadata=metadata,
        )

        return {
            "status": "success",
            "message": f"Archivo '{file.filename}' subido correctamente",
            "chunks_added": chunks_added,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error subiendo archivo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/stats", response_model=EmbeddingsStats, tags=["Documents"])
async def get_documents_stats():
    """Get statistics about the RAG knowledge base"""
    try:
        rag_service = get_rag_service()
        stats = rag_service.get_stats()

        return EmbeddingsStats(
            total_documents=stats.get("total_documents", 0),
            collection_name=stats.get("collection_name", ""),
            vector_size=stats.get("vector_size", 0),
        )
    except Exception as e:
        logger.error(f"Error geting documents statistics from RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/reset", tags=["Documents"])
async def reset_documents(is_admin: bool = Depends(check_admin_access)):
    """
    Reset the RAG knowledge base (delete all documents)
    WARNING: This will delete all uploaded documents!
    """
    try:
        rag_service = get_rag_service()
        rag_service.delete_collection()

        return {"status": "success", "message": "all documents have been deleted"}
    except Exception as e:
        logger.error(f"Error deleting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG
    )
