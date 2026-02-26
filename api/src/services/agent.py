"""
Agent service using LangGraph for intelligent
tool selection
"""

import json
import logging
from typing import Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# from langgraph.prebuilt import ToolExecutor, ToolInvocation
from src.core.config import settings
from src.core.models import ChatResponse, Source
from src.core.prompts import RESPONSE_SYSTEM_PROMPT, SYSTEM_PROMPT, make_response_prompt
from src.services.rag_store import get_rag_service
from src.services.tavily_search import get_tavily_service

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """state for agent graph"""

    messages: Sequence[BaseMessage]
    query: str
    tool_choice: Literal["rag", "tavily", "direct", "agent_decision", "multitool"]
    rag_results: list
    tavily_results: list
    final_response: str
    thought_process: str


class PortfolioAgent:
    """intelligent agent for routing querys"""

    def __init__(self) -> None:
        self.rag_service = get_rag_service()
        self.tavily_service = get_tavily_service()
        self.llm = None
        self._initialize_llm()
        self.graph = self._build_graph()

    def _initialize_llm(self):
        try:
            # use Open AI if this is provide in .env file
            if settings.OPENAI_API_KEY:
                self.llm = ChatOpenAI(
                    model=settings.OPENAI_MODEL,
                    temperature=0.7,
                    api_key=settings.OPENAI_API_KEY,
                )
                logger.info(f"using OpenAI LLM {settings.OPENAI_MODEL}")
            else:
                # or use OLLAMA if OPENAI_API_KEY is not provide in .env file
                self.llm = ChatOllama(
                    model=settings.OLLAMA_MODEL,
                    base_url=settings.OLLAMA_BASE_URL,
                    temperature=0.7,
                    timeout=60,
                )
                logger.info(f"Using OLLAMA LLM: {settings.OLLAMA_MODEL}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise

    def _decide_tool(self, state: AgentState) -> AgentState:
        """Decide which tool to use"""
        try:
            context = []
            if state.get("rag_results"):
                context.append("ya tienes resultados de RAG (CV/portafolio)")
            if state.get("tavily_results"):
                context.append("Ya tienes resultados de Tavily (web)")

            content = f"Pregunta: {state['query']}"
            if context:
                content += f"\n\nContexto: {' '.join(context)} ¿necesitas más información o ya puedes responder?"

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=content),
            ]

            response = self.llm.invoke(messages)

            # Parse JSON Response
            try:
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content.split("```json")[1].split("```")[0].strip()
                elif content.startswith("```"):
                    content = content.split("```")[1].split("```")[0].strip()

                decision = json.loads(content)

                if isinstance(decision, str):
                    tool_choice = (
                        decision
                        if decision in ("rag", "tavily", "direct", "final")
                        else "direct"
                    )
                    reasoning = ""
                else:
                    tool_choice = decision.get("tool", "direct")
                    reasoning = decision.get("reasoning", "")

                logger.info(f"Tool decision: {tool_choice}. {reasoning}")
                prev = state.get("thought_process", "")
                state["thought_process"] = (
                    f"{prev}• [{tool_choice}]: {reasoning}\n".strip()
                )
                # state["thought_process"] = reasoning

            except json.JSONDecodeError:
                # Fallback: simple heuristics
                query_lower = state["query"].lower()
                if any(
                    word in query_lower
                    for word in [
                        "experiencia",
                        "proyecto",
                        "habilidad",
                        "cv",
                        "educacion",
                        "trabajado",
                        "conocimiento",
                    ]
                ):
                    tool_choice = "rag"

                elif any(
                    word in query_lower
                    for word in [
                        "qué es",
                        "actualidad",
                        "noticia",
                        "busca",
                        "información sobre",
                    ]
                ):
                    tool_choice = "tavily"

                else:
                    tool_choice = "direct"
                logger.warning(f"JSON parsing failed, using heuristic: {tool_choice}")
                state["thought_process"] = f"Heurística: elegí {tool_choice}"

            state["tool_choice"] = tool_choice
            return state

        except Exception as e:
            logger.error(f"error to decide tool: {e}")
            state["tool_choice"] = "direct"
            state["thought_process"] = f"Error al decidir: {e}"
            return state

    def _router(self, state: AgentState) -> AgentState:
        """
        Decide which is the next node to execute with reference to state
        """
        tool = state.get("tool_choice", "direct")

        # If exist rag results
        if tool == "rag" and state.get("rag_results"):
            return "generate_response"
        # if exist tavily results
        if tool == "tavily" and state.get("tavily_results"):
            return "generate_response"

        if tool == "rag":
            return "execute_rag"
        if tool == "tavily":
            return "execute_tavily"

        return "generate_response"

    def _execute_rag(self, state: AgentState) -> AgentState:
        """Execute RAG search"""
        if state["tool_choice"] != "rag":
            return state

        try:
            sources = self.rag_service.search(state["query"])
            if sources:
                state["rag_results"] = [s.dict() for s in sources]
                logger.info(f"RAG search found {len(sources)} results")
            else:
                logger.info("RAG search found no results")
                state["rag_results"] = []
        except Exception as e:
            logger.error(f"error executing RAG: {e}")
            state["rag_results"] = []

        return state

    def _execute_tavily(self, state: AgentState) -> AgentState:
        """Execute Tavily Search"""
        if state["tool_choice"] != "tavily":
            return state

        try:
            if self.tavily_service.is_available():
                sources = self.tavily_service.search(state["query"])
                if sources:
                    state["tavily_results"] = [s.dict() for s in sources]
                    state["tool_choice"] = "tavily"
                logger.info(f"Tavily search found {len(sources)} results")
            else:
                state["tavily_results"] = []
                logger.warning("Tavily service not available")
        except Exception as e:
            logger.error(f"Error executing Tavily: {e}")
            state["tavily_results"] = []

        return state

    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate final response based on tool results"""
        try:
            rag_text = "\n".join(
                [
                    parcial_content["content"]
                    for parcial_content in state.get("rag_results", [])
                ]
            )
            web_text = "\n".join(
                [
                    parcial_content["content"]
                    for parcial_content in state.get("tavily_results", [])
                ]
            )

            # create a unificated prompt
            prompt_content = make_response_prompt(
                query=state["query"],
                tool_choice=state["tool_choice"],
                rag_context=rag_text,
                tavily_context=web_text,
            )

            # invoke llm for final redaction
            messages = [
                SystemMessage(content=RESPONSE_SYSTEM_PROMPT),
                HumanMessage(content=prompt_content),
            ]
            response = self.llm.invoke(messages)

            state["final_response"] = response.content

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state["final_response"] = (
                "Lo siento, hubo un error al procesar tu consulta."
            )

        return state

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("decide_tool", self._decide_tool)
        workflow.add_node("execute_rag", self._execute_rag)
        workflow.add_node("execute_tavily", self._execute_tavily)
        workflow.add_node("generate_response", self._generate_response)

        # Set entry point
        workflow.set_entry_point("decide_tool")

        # Add edges

        workflow.add_conditional_edges(
            "decide_tool",
            self._router,
            {
                "execute_rag": "execute_rag",
                "execute_tavily": "execute_tavily",
                "generate_response": "generate_response",
            },
        )
        workflow.add_edge("execute_rag", "decide_tool")
        workflow.add_edge("execute_tavily", "decide_tool")
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    def process_query(self, query: str, conversation_id: str) -> ChatResponse:
        """
        Process a user query through the agent

        Args:
            query: User query
            conversation_id: Conversation ID for tracking

        Returns:
            ChatResponse with the answer
        """
        try:
            # Initialize state
            initial_state = AgentState(
                messages=[HumanMessage(content=query)],
                query=query,
                tool_choice="direct",
                rag_results=[],
                tavily_results=[],
                final_response="",
                thought_process="",
            )

            # Run the graph
            final_state = self.graph.invoke(
                initial_state, config={"recursion_limit": 15}
            )

            # Prepare response
            # tool_used = final_state["tool_choice"]
            raw_tool = final_state["tool_choice"]
            rag = bool(final_state.get("rag_results"))
            tavily = bool(final_state.get("tavily_results"))

            if raw_tool == "final":
                tool_used = (
                    "multitool"
                    if (rag and tavily)
                    else "rag"
                    if rag
                    else "tavily"
                    if tavily
                    else "direct"
                )
            else:
                tool_used = "multitool" if (rag and tavily) else raw_tool

            # Build sources
            sources = []
            if final_state.get("rag_results"):
                sources.extend([Source(**r) for r in final_state["rag_results"]])
            if final_state.get("tavily_results"):
                sources.extend([Source(**r) for r in final_state["tavily_results"]])

            return ChatResponse(
                response=final_state["final_response"],
                conversation_id=conversation_id,
                tool_used=tool_used,
                sources=sources if sources else None,
                thought_process=final_state.get("thought_process") or None,
            )

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return ChatResponse(
                response="Lo siento, hubo un error al procesar tu consulta.",
                conversation_id=conversation_id,
                tool_used="direct",
                sources=None,
            )


# singleton instance
_agent = None


def get_agent() -> PortfolioAgent:
    """
    get or create agent singleton
    """
    global _agent
    if _agent is None:
        _agent = PortfolioAgent()
    return _agent
