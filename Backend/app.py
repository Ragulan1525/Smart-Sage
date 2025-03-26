from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
import ollama
import logging
import asyncio
import httpx
import os
import json
from dotenv import load_dotenv
import wikipedia
from ollama import chat
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from Backend.retrieval import (
    retrieve_relevant_text, process_uploaded_file, scrape_educational_websites,
    scrape_and_summarize, async_wikipedia_search, search_google, fetch_latest_news,store_text_in_chroma
)
from Backend.storage import store_text_in_chroma, get_chroma_collection

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ✅ Debugging: Print API keys (Remove in production)
if not GOOGLE_API_KEY or not GOOGLE_SEARCH_ENGINE_ID or not NEWS_API_KEY:
    raise ValueError("❌ Missing API keys! Check .env file.")

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# Load embedding model globally
model = SentenceTransformer('all-MiniLM-L6-v2')
logging.info("✅ Model loaded successfully!")

# Preload ChromaDB collection
chroma_collection = get_chroma_collection()

async def refresh_chroma():
    """Refresh ChromaDB by fetching updated web data for stored topics."""
    try:
        logging.info("🔄 Refreshing ChromaDB with updated web data...")

        # ✅ Retrieve all stored queries from ChromaDB
        stored_queries = chroma_collection.get(include=["documents"]).get("documents", [])

        # ✅ Refresh each stored query asynchronously
        for query in stored_queries:
            updated_google_data = await search_google(query)  # ✅ Correct way to call async function
            if updated_google_data:
                store_text_in_chroma("\n".join(updated_google_data), "Google Search (Refreshed)", model)

        logging.info("✅ ChromaDB refresh completed!")

    except Exception as e:
        logging.error(f"❌ Error during ChromaDB refresh: {e}")

# ✅ Schedule refresh every 24 hours using an async-friendly approach
scheduler = BackgroundScheduler()
scheduler.add_job(lambda: asyncio.create_task(refresh_chroma()), "interval", hours=24)
scheduler.start()

# ✅ Run refresh once at startup
asyncio.create_task(refresh_chroma())  # ✅ Proper way to call async function

class QueryRequest(BaseModel):
    question: str

async def async_wikipedia_search(query):
    """Fetch Wikipedia summary."""
    try:
        search_results = wikipedia.search(query)
        if search_results:
            return wikipedia.summary(search_results[0], sentences=2)
    except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError, Exception) as e:
        logging.error(f"❌ Wikipedia Error: {e}")
    return None

async def async_scrape_web(query):
    """Scrape educational websites asynchronously."""
    return scrape_educational_websites(query, model) or None

async def search_google(query):
    """Fetch live search results using Google Custom Search API."""
    try:
        async with httpx.AsyncClient() as client:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "q": query,
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_SEARCH_ENGINE_ID,
                "num": 5,
            }
            response = await client.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [f"{item['title']}: {item['link']}" for item in data.get("items", [])] if data.get("items") else None
    except Exception as e:
        logging.error(f"❌ Google Search API Error: {e}")
        return None

async def fetch_latest_news(query):
    """Fetch live news using NewsAPI."""
    try:
        async with httpx.AsyncClient() as client:
            url = f"https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "apiKey": NEWS_API_KEY,
                "language": "en",
                "pageSize": 5,
            }
            response = await client.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [f"{article['title']}: {article['url']}" for article in data.get("articles", [])] if data.get("articles") else None
    except Exception as e:
        logging.error(f"❌ News API Error: {e}")
        return None

@app.post("/ask")
async def answer_question(request: QueryRequest):
    """Handles user queries using a Hybrid Retrieval Model with LLM-based classification."""
    try:
        query = request.query.strip()
        context_parts = []

         # Personalized greeting
        if "hello" in query.lower() or "hi" in query.lower():
            return {"answer": "👋 Hello! I'm here to help you with your study questions. What would you like to know today?"}
        

        # ✅ Step 1: Ask LLM if the Query is Educational
        classification_prompt = f"""
        You are an AI that classifies queries as **educational** or **non-educational**.
        If the query is not educational, suggest a related **educational topic** instead.

        Example:
        - Query: "Who is the hero in KGF?"
          Classification: Non-educational. Suggested Educational Topic: "Impact of Cinema on Indian Culture"
        
        - Query: "What is a doubly linked list?"
          Classification: Educational. Suggested Educational Topic: "None"

        Now classify the following query:

        Query: "{query}"

        Respond in JSON format:
        {{"classification": "Educational" or "Non-Educational", "suggested_topic": "..." or "None"}}
        """

        classification_response = chat(model="mistral", messages=[
            {"role": "system", "content": "You are a classifier for educational queries."},
            {"role": "user", "content": classification_prompt}
        ])
        
        classification_result = classification_response.get('message', {}).get('content', "")
        logging.info(f"🔍 LLM Classification: {classification_result}")

        # ✅ Parse classification response
        try:
            classification_data = json.loads(classification_result)
            is_educational = classification_data.get("classification", "Non-Educational") == "Educational"
            suggested_topic = classification_data.get("suggested_topic", None)
        except json.JSONDecodeError:
            logging.error("❌ LLM classification failed, assuming non-educational query.")
            is_educational, suggested_topic = False, None

        # ✅ Step 2: Modify Query if Non-Educational
        if not is_educational:
            if suggested_topic:
                query = suggested_topic  # Replace with suggested educational topic
                user_message = f"⚠️ Your original question was non-educational. How about we explore this related topic: **{query}**? 📚"

            else:
                return {"answer": "⚠️ This question isn't educational. Can you ask something related to your studies? I'm here to help!"}
        else:
            user_message = None  # No need to inform the user

        # ✅ Step 3: Retrieve Data from All Sources
        logging.info(f"📩 Received query: {query}")

        query = query.lower().strip()
    
        chroma_result = await retrieve_relevant_text(query, model)
        logging.info(f"🔎 ChromaDB Result: {chroma_result}")

        wiki_result = await async_wikipedia_search(query)
        logging.info(f"📖 Wikipedia Result: {wiki_result}")

        web_scrape_result = await scrape_educational_websites(query, model)
        logging.info(f"🌐 Web Scrape Result: {web_scrape_result}")

        news_result = await fetch_latest_news(query)
        logging.info(f"📰 NewsAPI Result: {news_result}")

        google_result = await search_google(query)
        logging.info(f"🔍 Google Search Result: {google_result}")


        # ✅ Extract only URLs (assuming search_google returns a list of dictionaries with 'link' key)
        web_scrape_urls = [result['link'] for result in search_google if 'link' in result]

        # ✅ Web Scraping & Summarization
        web_scrape_results = await asyncio.gather(*[scrape_and_summarize(url) for url in web_scrape_urls])
        web_scrape_results = [result for result in web_scrape_results if result]  # Remove None values


        if chroma_result:
            context_parts.append(chroma_result)
            return {"response": chroma_result}
        else:
            if wiki_result:
                store_text_in_chroma(wiki_result, "Wikipedia")
                context_parts.append(wiki_result)
                return {"response": wiki_result}
            if web_scrape_results:
                combined_web_data = "\n".join(web_scrape_results)
                store_text_in_chroma(combined_web_data, "Web Scraping")
                context_parts.append(combined_web_data)
                return {"response": "\n".join(web_scrape_result)}
            if google_result:
                combined_google_data = "\n".join(google_result)
                store_text_in_chroma(combined_google_data, "Google Search")
                context_parts.append(combined_google_data)
                return {"response": "\n".join(google_result)}
            if news_result:
                combined_news_data = "\n".join(news_result)
                store_text_in_chroma(combined_news_data, "News API")
                context_parts.append(combined_news_data)
                return {"response": "\n".join(news_result)}
                

        # ✅ Step 5: Prepare Context for LLM
        context = "\n".join(context_parts) if context_parts else None

        if not context or all(link in context for link in context_parts):
           logging.warning("⚠️ No relevant data found in ChromaDB or Web Sources.")

           # Handling date-based queries
           if re.match(r"\d{1,2}\s\w+\s\d{4}", query):  # Matches "24 March 2025"
             return {"answer": "I'm designed to provide educational content. Could you ask something related to your studies?"}

           # Manually inject an educational answer
           if "developer of python" in query.lower():
             return {"answer": "Python was developed by Guido van Rossum in 1991."}
    
           # If no known fallback, ask user for a study-related query
           return {"answer": "I'm here to provide educational information. Could you please ask something related to your studies?"}


        logging.info(f"📌 Context sent to LLM: {context[:500]}...")

        # ✅ Step 6: Ask LLM to Generate an Answer Based on Retrieved Context
        llm_prompt = f"""
        You are an AI tutor specializing in educational topics.
        Answer the following question based on the retrieved context.
        If no relevant data is found, generate an educational response.

        Question: {query}

        Context: {context}

        Ensure the response is:
        - **Relevant to the query** (Avoid unrelated topics)
        - **Accurate and informative**
        - **Concise yet detailed for study purposes**
        
- If the question is about movies, provide insights into film-making, cinematography, or storytelling techniques.
- If the question is general knowledge, provide an informative, research-based answer.
- If the question is not related to education, politely ask the user to ask something else.
"""
    
        

        llm_response = chat(model="mistral", messages=[
            {"role": "system", "content": "You are an AI tutor. Use the provided context strictly."},
            {"role": "user", "content": llm_prompt}
        ])
        llm_answer = llm_response.get('message', {}).get('content', "")
        

        # ✅ Step 7: Return Answer (with user message if modified)
        final_answer = f"{user_message}\n\n{llm_answer}" if user_message else llm_answer
        final_answer += "\n\n🤔 If you have more questions or need clarification, feel free to ask!"
        return {"answer": final_answer or "❌ Could not generate a response."}

    except Exception as e:
        logging.error(f"❌ Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")



# ✅ Load summarizer (Hugging Face)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

COMMON_RESPONSES = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi there! What would you like to learn about?",
    "hey": "Hey! Feel free to ask me anything.",
}


@app.get("/query")
async def get_answer(query: str):
    """Handles user queries and retrieves relevant information."""
    logging.info(f"Received query: {query}")

    query = query.lower().strip()
    if query in COMMON_RESPONSES:
        return {"response": COMMON_RESPONSES[query]}

    if len(query) < 3:  
        return {"response": "Could you provide more details?"}

    chroma_result = await retrieve_relevant_text(query, model)
    wiki_result = await async_wikipedia_search(query)
    google_result = await search_google(query)
    web_scrape_result = await scrape_and_summarize(query)

    response_parts = []
    if chroma_result:
        response_parts.append(f"📚 **ChromaDB Result:**\n{chroma_result}")
    if wiki_result:
        response_parts.append(f"🌍 **Wikipedia:**\n{wiki_result}")
    if google_result:
        response_parts.append(f"🔎 **Google Results:**\n" + "\n".join(google_result))
    if web_scrape_result:
        response_parts.append(f"📄 **Extracted Summary:**\n{web_scrape_result}")

    if response_parts:
        return {"response": "\n\n".join(response_parts)}

    return {"response": "❌ No relevant data found. Try asking a different educational question. 😊"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handles file uploads and stores the extracted content in ChromaDB."""
    try:
        text = process_uploaded_file(file, model)
        logging.info(f"✅ File '{file.filename}' processed and stored successfully.")
        return {"message": "✅ File processed and stored successfully"}
    except Exception as e:
        logging.error(f"❌ Error processing file upload: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the file.")




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
