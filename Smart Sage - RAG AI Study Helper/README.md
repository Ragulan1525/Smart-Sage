Smart Sage - AI Study Helper Chatbot
A smart AI-powered chatbot designed to assist users with educational queries using Retrieval-Augmented Generation (RAG). It integrates ChromaDB, web scraping, Wikipedia search, and file processing (PDF, DOCX, TXT) to provide accurate study-related responses.

🚀 Features
✅ Education-Focused Responses – Filters out non-educational queries.
✅ ChromaDB Integration – Stores and retrieves study-related content for enhanced accuracy.
✅ Web Scraping & Wikipedia Search – Fetches relevant educational content dynamically.
✅ File Upload Support – Extracts content from PDFs, DOCX, and TXT files for personalized assistance.
✅ FastAPI Backend – Efficient and scalable API implementation.

🛠 Tech Stack
Backend: FastAPI
Database: ChromaDB (Vector Database for RAG)
AI Models: Ollama (Llama 3), SentenceTransformers
Web Scraping: BeautifulSoup, Wikipedia API
File Processing: PyMuPDF, python-docx
Deployment: Uvicorn

Installation & Setup
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/AI-Study-Helper-Chatbot.git
cd AI-Study-Helper-Chatbot
2️⃣ Create a Virtual Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Run the Chatbot API
bash
Copy
Edit
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
5️⃣ Access the API
Open: http://localhost:8000/docs
Test the API using Swagger UI

API Endpoints
1️⃣ Ask a Question
Endpoint: POST /ask
Request:
json
Copy
Edit
{
  "question": "What is machine learning?"
}
Response:
json
Copy
Edit
{
  "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve their performance without explicit programming."
}
2️⃣ Upload a Study File (PDF/DOCX/TXT)
Endpoint: POST /upload
Request: Upload a file
Response:
json
Copy
Edit
{
  "message": "File processed successfully",
  "extracted_text": "Content from the uploaded file..."
}

How It Works (RAG Architecture)
User asks a question.
The chatbot searches ChromaDB for relevant stored knowledge.
If no relevant data is found, it searches Wikipedia and educational websites.
If no online sources match, it generates a response using Ollama's LLM.
Stores useful responses in ChromaDB for future retrieval.

Conclusion
The AI Study Helper Chatbot is a powerful tool designed to enhance learning by providing accurate, relevant, and interactive educational support. By combining retrieval-based knowledge with AI-generated responses, it ensures that users receive high-quality study assistance. Future enhancements could include multi-language support, more advanced AI models, and integration with additional learning platforms.

This project demonstrates the potential of AI in education and how RAG-based chatbots can revolutionize the way students and educators access information.

