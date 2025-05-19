from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from Backend.retrieval import retrieve_relevant_text


app = FastAPI()

logging.basicConfig(level=logging.DEBUG)

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def answer_question(request: QueryRequest):
    """Handles user queries and provides responses."""
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        # Example: Retrieve relevant text from ChromaDB
        context = await retrieve_relevant_text(question, model=None, top_k=3)  # Pass the correct model if needed
        if context and "No relevant results found" not in context:
            return {"answer": context}

        # If no context is found, fallback to a default response
        return {"answer": "I couldn't find relevant information. Try rephrasing your question!"}

    except HTTPException as http_exc:
        logging.error(f"HTTP error: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again later.")