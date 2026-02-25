"""
tavily search is used for general querys these
cannot be answered by the documents stored in RAG Services
"""

import logging
from typing import List

from src.core.config import settings
from src.core.models import Source
from tavily import TavilyClient

logger = logging.getLogger(__name__)


class TavilyService:
    """
    Service for web search using tavily
    """

    def __init__(self) -> None:
        self.client = None
        self._initialize()

    def _initialize(self):
        """Initialize Tavily client"""

        try:
            if not settings.TAVILY_API_KEY:
                logger.warning("Tavily API key not configures")
                return

            self.client = TavilyClient(api_key=settings.TAVILY_API_KEY)
            logger.info("Tavily service initilized succesfully")
        except Exception as e:
            logger.error(f"Error initializing Tavily service {e}")
            self.client = None

    def search(
        self, query: str, max_results: int = 5, search_depth: str = "basic"
    ) -> List[Source]:
        """
        Search the web using Tavily

        Args:
            query: search query
            max_results : maximum number of results
            search_depth: "basic" or "advanced"

        Returns:
            List of Source objects with search results
        """

        if not self.client:
            logger.warning("Tavily client not initialized")
            return []

        try:
            # perform search
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=True,
                include_raw_content=False,
            )

            sources = []

            # add the AI generated answer if available
            if response.get("answer"):
                sources.append(
                    Source(
                        content=response["answer"],
                        source="Tavily AI summary",
                        relevance_score=1.0,
                    )
                )

            # Add individual search results
            for results in response.get("results", []):
                source = Source(
                    content=results.get("content", ""),
                    source=results.get("url", "Unknown"),
                    relevance_score=results.get("score", 0.0),
                )
                sources.append(source)

            logger.info(
                f"successfull search in web with tavily. Found {len(sources)} results from Tavily."
            )
            return sources
        except Exception as e:
            logger.error(f"error to search data in web with Tavily: {e}")
            return []

    def get_answer(self, query: str) -> str:
        """
        Get a direct answer from tavily

        Args:
            query: search query

        Returns:
            Direct answer string.
        """

        if not self.client:
            return "search service nor available"

        try:
            response = self.client.search(
                query=query, max_results=3, search_depth="basic", include_answer=True
            )

            return response.get("answer", "No answer found")

        except Exception as e:
            logger.error(f"error getting answer from tavily: {e}")
            return "Error retrieving answer"

    def is_available(self) -> bool:
        """
        check if Tavily sercice is available
        """
        return self.client is not None


# singleton instance
_tavily_service = None


def get_tavily_service() -> TavilyService:
    """Get or create Tavily service singleton"""
    global _tavily_service
    if _tavily_service is None:
        _tavily_service = TavilyService()
    return _tavily_service
