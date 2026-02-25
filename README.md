# 🎯 Chatbot Portfolio - Project Summary

## 📋 Overview

This is a **complete intelligent chatbot system** for your personal portfolio that combines:

- **RAG (Retrieval Augmented Generation)** with Qdrant to answer questions about your resume and experience
- **Tavily Search API** for web searches on general topics
- **LangGraph Agent** that intelligently decides which tool to use
- **FastAPI** as a robust and scalable backend
- **Modern web interface** ready to use

## 🏗️ Implemented Architecture

```
┌─────────┐
│ Usuario │
└────┬────┘
     │
     ▼
┌─────────────────┐
│ Frontend HTML   │
│ (Chat UI)       │
└────┬────────────┘
     │
     ▼
┌──────────────────────────────────────────────┐
│           FastAPI Gateway                    │
│  • /chat - Endpoint principal                │
│  • /documents/* - Gestión de documentos      │
│  • /health - Estado del sistema              │
└────┬─────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────┐
│         LangGraph Agent                      │
│  Decide automáticamente:                     │
│  • ¿Es sobre el portafolio? → RAG            │
│  • ¿Es búsqueda general? → Tavily            │
│  • ¿Es multitool? → RAG + Tavily             │
│  • ¿Es conversación? → Respuesta directa     │
└────┬─────────────────────────────────────────┘
     │
     ├─────────────┬─────────────┬──────────────┐
     ▼             ▼             ▼              ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│   RAG   │  │ Tavily  │  │  Direct │  │   LLM   │
│ Service │  │ Search  │  │Response │  │ (Ollama/│
│         │  │         │  │         │  │ OpenAI) │
└────┬────┘  └─────────┘  └─────────┘  └─────────┘
     │
     ▼
┌─────────────┐
│   Qdrant    │
│ Vector DB   │
│ (embeddings)│
└─────────────┘
```

## ✨ Key Features

### 1. **Intelligent Agent with LangGraph**
- Analyzes each user question
- Automatically decides on the best strategy
- Uses multiple tools as needed

### 2. **Complete RAG System**
- Vector database with Qdrant
- Optimized embeddings (all-MiniLM-L6-v2)
- Intelligent document chunking
- Advanced semantic search

### 3. **Integration with Tavily**
- Real-time web searches
- Results with verified sources
- Context-enriched responses

### 4. **Complete REST API**
- Automatic documentation (Swagger)
- Endpoints for all operations
- Robust error handling
- CORS configured

### 5. **User Interface**
- Modern and responsive chat
- Visual tool indicators
- Displays information sources
- Animations and polished UX

## 🔑 Configuración Necesaria

### Required API Keys

1. **Tavily API** (Recommended - Free)
   - Sign up at: https://tavily.com
   - Get a free API key
   - Add it to `.env`: `TAVILY_API_KEY=“tvly-xxxxx”`

2. **LLM - Choose an option:**
   
   **Option A: Ollama (Local - Free)**
   ```bash
   # Install from https://ollama.ai
   ollama pull llama3.2
   ```
   
   **Option B: OpenAI (Cloud - Paid)**
   ```env
   OPENAI_API_KEY="sk-xxxxx"
   ```

### Environment Variables (.env)

```env
# Web search
TAVILY_API_KEY="tvly-xxxxx"

# LLM (choose one)
OLLAMA_MODEL="llama3.2"              # If you use Ollama
# OPENAI_API_KEY="sk-xxxxx"         # If you use OpenAI

# Vector DB
QDRANT_URL="http://localhost:6333"

# RAG Configuration
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=3
```

## 🧪 Test the System

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Test Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H “Content-Type: application/json” \
  -d ‘{“message”: “What experience do you have with Python?”}’
```

### 3. Sample Questions

**About the portfolio (use RAG):**
- “What experience do you have?”
- “What are your projects?”
- “What skills do you have?”
- “Tell me about your education.”

**General searches (use Tavily):**
- “What is machine learning?”
- “What is the latest AI news?”
- “How does FastAPI work?”

**Casual conversation (direct response):**
- “Hello”
- “Thanks for the information”
- “How are you?”

## 🎯 Use Cases

### 1. Personal Portfolio
- Answer questions about your experience
- Showcase relevant projects
- Explain your technical skills

### 2. Interactive CV
- Recruiters can chat with your CV
- Semantic search for experiences
- Context enriched with examples

### 3. General Assistant
- Answer technical questions with web search
- Combine your knowledge with current information
- Natural and fluid conversation

## 🔧 Customization

### 1. Add Your Information
Edit `http://localhost:8000/documents/upload_file` with (better .md files):
- Your work experience
- Projects completed
- Technical skills
- Education and certifications

### 2. Adjust Agent Behavior
Edit `src/services/agent.py`:
```python
SYSTEM_PROMPT = “”“Your custom prompt here...”“”
```

### 3. Change Embeddings Model
Edit `src/services/rag_store.py`:
```python
model_name="sentence-transformers/all-MiniLM-L6-v2"
```

Options:
- `all-MiniLM-L6-v2` (384 dim) - Fast
- `all-mpnet-base-v2` (768 dim) - Better quality
- `paraphrase-multilingual-MiniLM-L12-v2` - Better Spanish

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | General Info       |
| `/health` | GET | System Status |
| `/chat` | POST | Main Chat  |
| `/chat/auth` | POST | Chat Authentication |
| `/documents/upload_file` | POST | Upload document (pdf, txt, md) token required |
| `/documents/stats` | GET | DB Statistics |
| `/documents/reset` | DELETE | Reset DB token required |

Complete documentation: `http://localhost:8000/docs`

## 🐛 Troubleshooting

### Error: “Docker connection refused”
```bash
docker-compose restart
```

### Error: “Module not found”
```bash
pip install -r requirements.txt
```

### Ollama does not connect
- Check: `ollama list`
- Or use OpenAI instead

## 📈 Metrics and Monitoring

```bash
# View statistics
curl http://localhost:8000/documents/stats

# Server logs
# Displayed in the terminal where you ran main.py

# Docker logs
docker-compose logs -f
```

## 🚢 Production Deployment

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

## 📚 Additional Resources

- **LangChain Docs**: https://python.langchain.com
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Qdrant Docs**: https://qdrant.tech/documentation/
- **Tavily API**: https://docs.tavily.com
- **FastAPI Docs**: https://fastapi.tiangolo.com

## 🎓 Next Steps

1. **Customize with your information**
   - Edit `populate_db.py`
   - Run to update the DB

2. **Configure API keys**
   - Tavily (for web searches)
   - OpenAI (optional, for better LLM)

3. **Adjust agent prompts**
   - Customize behavior
   - Add specific logic

4. **Improve frontend**
   - Customize colors/design
   - Add your branding
   - Deploy to your domain

5. **Expand functionality**
   - Integrate more data sources
   - Add authentication
   - Implement analytics

## 💡 Suggested Future Improvements

- [ ] Conversation persistence
- [ ] Multiple languages
- [ ] Google Drive integration
- [ ] WebSocket for streaming
- [ ] Metrics and analytics
- [ ] A/B testing of prompts
- [ ] Rate limiting
- [ ] JWT authentication

## 📝 Important Notes

1. **Privacy**: Data is stored locally on Qdrant
2. **Costs**: Ollama is free, OpenAI has usage costs
3. **Performance**: With local Ollama, it may be slower
4. **Scalability**: For production, consider Qdrant cloud

## 🤝 Soporte

Si encuentras problemas:
1. Revisa el `README.md` completo
2. Consulta `QUICKSTART.md`
3. Ejecuta `python test_chatbot.py`
4. Verifica logs del servidor

## 📄 Licencia

MIT License - Usa libremente para tu portafolio

---

**¡Éxito con tu Portfolio Chatbot! 🚀**

Para cualquier duda, consulta la documentación o los comentarios en el código.
