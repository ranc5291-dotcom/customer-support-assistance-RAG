
🚀 AI Customer Support Assistant

📌 Project Description
AI-powered customer support system
Uses RAG (Retrieval-Augmented Generation)
Integrated with LLM for intelligent responses
Provides context-aware answers using real data
Supports human escalation when needed

🎯 Objective
Automate customer support using AI
Improve response accuracy using knowledge base
Reduce manual effort
Provide fallback to human support

⚙️ Tech Stack
Backend: FastAPI
Frontend: HTML, CSS, JavaScript
AI/LLM: Groq API
Database: ChromaDB (Vector Database)
Architecture: RAG
Voice: Browser Speech Recognition
Frameworks: LangChain, LangGraph

🧠 System Architecture
User sends query
System performs intent detection
Query is routed to one of three paths:
Knowledge Path → RAG + LLM
Simple Reply → Predefined responses
Escalation → Admin/Human support

🔄 Workflow
🔹 Ingestion Flow
Upload documents (PDF/TXT)
Split documents into chunks
Convert chunks into embeddings
Store embeddings in vector database

🔹 Query Flow
User sends query
Detect user intent
Retrieve relevant chunks
Send data to LLM
Generate response
Escalate if required

✨ Features
RAG-based response generation
Smart query routing
Voice-enabled interaction
Admin dashboard
Chat history tracking
Human-in-the-loop escalation


📊 Admin Panel
Upload documents
Delete documents
Manage knowledge base
View user queries
Monitor AI responses
Track sessions and analytics
Handle escalation tickets


📂 Project Structure
backend/ → FastAPI backend
frontend/ → User interface
admin-panel/ → Admin dashboard
data/ → Uploaded documents
main.py → Entry point
requirements.txt → Dependencies
README.md → Documentation

🚀 Installation & Setup
Clone repository
git clone https://github.com/ranc5291-dotcom/customer-support-assistance-RAG
cd customer-support-assistance-RAG
Create virtual environment
python -m venv venv
venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
Setup environment variables
Create .env file
Add:
GROQ_API_KEY=your_api_key_here
Run application
uvicorn main:app --reload


🎯 Use Cases
Customer support automation
AI chatbots
Knowledge-based assistants
Internal helpdesk systems


🔐 Security
API keys stored using environment variables
Sensitive data not exposed in repository

🚧 Future Improvements
Multi-language support
UI/UX improvements
Advanced analytics
Third-party integrations
