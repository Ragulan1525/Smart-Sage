from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
import ollama
import logging
import asyncio
import httpx
import os
import json
import validators
from dotenv import load_dotenv
from chromadb import PersistentClient
import wikipedia
from ollama import chat
from textwrap import shorten
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from apscheduler.schedulers.background import BackgroundScheduler
from Backend.mongodb_utils import save_message, get_chat_history
from uuid import uuid4
from fastapi import Query
from docx import Document
from chromadb import PersistentClient
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from Backend.retrieval import (
    retrieve_relevant_text, process_uploaded_file, scrape_educational_websites,
    scrape_and_summarize, async_wikipedia_search, search_google, fetch_latest_news,store_text_in_chroma,search_duckduckgo
)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Backend")))

from Backend.storage import store_text_in_chroma_simple, get_chroma_collection

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



@app.on_event("startup")
async def startup_event():
    asyncio.create_task(refresh_chroma())

async def refresh_chroma():
    """Refresh ChromaDB by fetching updated web data for stored topics."""
    try:
        logging.info("🔄 Refreshing ChromaDB with updated web data...")

        # ✅ Retrieve all stored queries from ChromaDB
        stored_queries = chroma_collection.get(include=["documents"]).get("documents", [])

        # ✅ Refresh each stored query asynchronously
        for query in stored_queries:
            updated_google_data = await search_google(query,model)  # ✅ Correct way to call async function
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
    """Fetch Wikipedia summary if relevant to query."""
    try:
        search_results = wikipedia.search(query, results=3)
        for result in search_results:
            summary = wikipedia.summary(result, sentences=3)
            if "linked list" in summary.lower():  # Ensure relevance
                return summary
    except Exception as e:
        logging.error(f"Error fetching Wikipedia data: {e}")
    return None


async def async_scrape_web(query):
    """Scrape educational websites asynchronously."""
    return scrape_educational_websites(query, model) or None


import logging
import asyncio
import httpx
import validators
import random

class RateLimitError(Exception):
    pass

async def search_google(query, model):
    """Fetch live search results from Google, fallback to DuckDuckGo if Google fails."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_SEARCH_ENGINE_ID,
        "num": 3,  # Number of results to fetch
    }

    retries = 3  # Number of retry attempts

    async with httpx.AsyncClient() as client:
        for attempt in range(retries):
            try:
                response = await client.get(url, params=params, timeout=10)
                response.raise_for_status()

                if response.status_code == 200:
                    data = response.json()
                    if "items" in data and data["items"]:
                        return await process_search_results(data, query, model)
                    else:
                        logging.warning("⚠️ No results from Google, trying DuckDuckGo...")
                        return await search_duckduckgo(query, model)  # Fallback to DuckDuckGo

            except httpx.RequestError as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logging.warning(f"⚠️ Network error: {e}, Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                   wait_time = random.uniform(5, 10)  # Randomized delay to avoid detection
                   logging.warning(f"⚠️ Google API Rate Limit. Retrying in {wait_time:.1f} seconds...")
                   await asyncio.sleep(wait_time)

                else:
                    logging.error(f"❌ Google API Error {e.response.status_code}: {e.response.text}")
                    break

            await asyncio.sleep(5)  # Wait before retrying
    try:
       result = await search_google(...)
    except RateLimitError:
       logging.warning("⚠️ Google failed. Switching to DuckDuckGo after 1 attempt.")
       result = await search_duckduckgo(...)

    logging.warning("⚠️ Google search failed, switching to DuckDuckGo...")
    return await search_duckduckgo(query, model)  # Fallback if Google fails


async def search_duckduckgo(query, model):
    """Fetch search results from DuckDuckGo API (No API key required)."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

        logging.debug(f"Raw DuckDuckGo response for query '{query}': {data}")

        # Extract the main abstract text or fallback to related topics
        main_text = data.get("AbstractText", "").strip()
        related_topics = [t["Text"] for t in data.get("RelatedTopics", []) if "Text" in t]

        # Combine results if available
        extracted_results = [main_text] if main_text else related_topics

        if extracted_results:
            text_data = "\n".join(extracted_results)
            store_text_in_chroma(text_data, f"DuckDuckGo Data ({query})", model)
            return extracted_results

        logging.warning(f"⚠️ No relevant results found for '{query}'.")
        return ["No valid results found."]

    except httpx.RequestError as e:
        logging.error(f"❌ HTTP Request error while fetching DuckDuckGo data: {e}")
    except Exception as e:
        logging.error(f"❌ Unexpected error in search_duckduckgo: {e}")

    return ["No valid results found."]  # Explicit return on failure


async def process_search_results(data, query, model):
    """Process search results from Google API and extract relevant information."""
    extracted_results = []
    
    for item in data.get("items", []):
        link = item.get("link")
        if link and validators.url(link):  # Validate URL before using
            page_text = await scrape_and_summarize(link)
            extracted_results.append(f"{item['title']}: {page_text}" if page_text else f"{item['title']}: {link}")

    if extracted_results:
        store_text_in_chroma("\n".join(extracted_results), f"Google Data ({query})", model)
        return extracted_results
    
    logging.warning("⚠️ Google returned results, but none were usable. Falling back to DuckDuckGo.")
    return await search_duckduckgo(query, model)  # Fallback if Google data isn't useful


async def fetch_latest_news(query, date=None):
    """Fetch live news using NewsAPI."""
    try:
        async with httpx.AsyncClient() as client:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "apiKey": NEWS_API_KEY,
                "language": "en",
                "pageSize": 5,
                "sortBy": "relevancy"
            }
            if date:
                # Format date as YYYY-MM-DD
                formatted = date.strftime("%Y-%m-%d")
                params["from"] = formatted
                params["to"] = formatted

            response = await client.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return [f"{article['title']}: {article['url']}" for article in data.get("articles", [])] if data.get("articles") else None

    except Exception as e:
        logging.error(f"❌ News API Error: {e}")
        return None




app = FastAPI()

@app.post("/ask")
async def answer_question(request: QueryRequest):
    """Handles user queries using a Hybrid Retrieval Model with LLM-based classification."""
    try:
        query = request.query.strip()
        context_parts = []

        # ✅ Personalized Greeting
        if any(word in query.lower() for word in ["hello", "hi"]):
            return {"answer": "👋 Hello! I'm here to help with your study questions. What would you like to know?"}

        # ✅ Step 1: Classify Query as Educational or Non-Educational
        classification_prompt = f"""
        You are an AI that classifies queries as **educational** or **non-educational**.
        If non-educational, suggest a related **educational topic**.

        Query: "{query}"
        Respond in JSON format:
        {{"classification": "Educational" or "Non-Educational", "suggested_topic": "..." or "None"}}
        """

         # ✅ Ensure chat method supports async (Handling LLM Response Issue)
        try:
            classification_response = await asyncio.to_thread(ollama.chat, model="mistral", messages=[
                {"role": "system", "content": "You are an AI classifier that determines if a query is educational. If the query is educational, return 'educational'. Otherwise, return 'non-educational'."},
                {"role": "user", "content": classification_prompt}
            ])
            logging.info(f"🔍 LLM Classification Response: {classification_response}")
        except Exception as e:
            logging.error(f"❌ Error in LLM classification: {e}")
            raise HTTPException(status_code=500, detail="Error in classification using LLM.")


        # ✅ Proper JSON Extraction from LLM Response
        classification_text = classification_response.get('message', {}).get('content', "").strip()



        try:
          classification_data = json.loads(classification_response.get('message', {}).get('content', "").strip())
          is_educational = classification_data.get("classification") == "Educational"
          suggested_topic = classification_data.get("suggested_topic", None)
        except json.JSONDecodeError:
          logging.error("❌ JSON decoding error, assuming non-educational.")
          is_educational, suggested_topic = False, None



        # ✅ Modify Query if Non-Educational
        if not is_educational:
            if suggested_topic:
                query = suggested_topic
                user_message = f"⚠️ Your question was non-educational. How about exploring this topic: **{query}**? 📚"
            else:
                return {"answer": "⚠️ This question isn't educational. Please ask something related to your studies!"}
        else:
            user_message = None

        logging.info(f"📩 Processing query: {query}")

        # ✅ Step 2: Retrieve Data from Multiple Sources
        retrieval_tasks = [
            retrieve_relevant_text(query, model),
            async_wikipedia_search(query),
            scrape_educational_websites(query, model),
            async_scrape_web(query),    
            fetch_latest_news(query),
            search_google(query)
        ]
    
        
        results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

        # ✅ Handling Retrieval Errors & Filtering Valid Data
        chroma_result, wiki_result, web_scrape_result, news_result, google_result = [
            result if not isinstance(result, Exception) else None for result in results
        ]

        # ✅ Extract and Summarize Web Data
        web_scrape_urls = [item['link'] for item in google_result if 'link' in item] if google_result else []
        web_scrape_summaries = await asyncio.gather(*[scrape_and_summarize(url) for url in web_scrape_urls])
        web_scrape_summaries = [summary for summary in web_scrape_summaries if summary]

        doc_id = "doubly_linked_list_1"
        existing_docs = chroma_collection.get(ids=[doc_id])

        if not existing_docs["ids"]:  # Only add if it doesn't exist
          chroma_collection.add(
            documents=["A doubly linked list is a type of linked list where each node contains a data part and two pointers, one pointing to the previous node and the other pointing to the next node."], 
            metadatas=[{"source": "manual_data"}], 
            ids=[doc_id]
          )
    
    

        # ✅ Store and Return Data from Sources
        context_parts = []

        if chroma_result:
          context_parts.append(chroma_result)

        if wiki_result:
          #wiki_summary = wiki_result.strip()  # Ensure no extra spaces
          store_text_in_chroma(wiki_result, "Wikipedia", model)
          context_parts.append(wiki_result)

        if web_scrape_summaries:
           combined_web_data = "\n".join(web_scrape_summaries)
           store_text_in_chroma(combined_web_data, "Web Scraping", model)
           context_parts.append(combined_web_data)

        
        # if google_result:
        #    google_texts = [
        #       item.get("title", "") + " - " + item.get("snippet", "")  # Keep title + snippet
        #       for item in google_result
        #       if "snippet" in item
        #     ]

        #    if google_texts:  # Store only if valid results exist
        #      combined_google_data = "\n".join(google_texts)
        #      store_text_in_chroma(combined_google_data, "Google Search", model)

        #    context_parts.extend(google_texts)
        # else:
        #     google_texts = []

        if google_result:
            google_summaries = [item.get('snippet', '') for item in google_result if "snippet" in item]
            if google_summaries:
                combined_google_data = "\n".join(google_summaries)
                store_text_in_chroma(combined_google_data, "Google Search", model)
                # Instead of appending the raw data, we now add the summaries.
                context_parts.extend(google_summaries)


        if news_result:
           news_summaries = [item.strip() for item in news_result if item]  # Remove empty entries
           if news_summaries:  # Store only if there are valid results
             combined_news_data = "\n".join(news_summaries)
             store_text_in_chroma(combined_news_data, "News API", model)
           context_parts.append(combined_news_data)



        # ✅ Step 3: Check if We Have Any Valid Data
        context = "\n".join(context_parts) if context_parts else None

        if not context:
           logging.warning("⚠️ No relevant data found. Using LLM fallback.")
  
    # ✅ Directly Generate Answer from LLM
           llm_fallback_prompt = f"""
You are an AI tutor with vast knowledge.
The user asked: "{query}"
No relevant data was found from ChromaDB, Wikipedia, or Google.

💡 **Directly answer this question** as best as you can.
- If it's about data structures, explain it in detail.
- If it's about AI, programming, or tech, give a well-structured answer.
- If it's general knowledge, provide an educational response.
- Never say "I don't know". Always try to generate an answer.
"""

           logging.info("📌 Attempting LLM Fallback for query: %s", query)
    
           llm_fallback_response = await chat(model="mistral", messages=[
             {"role": "system", "content": "You are an AI tutor. If no context is provided, generate an educational answer."},
             {"role": "user", "content": llm_fallback_prompt}
            ])
    

           llm_fallback_answer = llm_fallback_response.get('message', {}).get('content', "").strip() if llm_fallback_response else "I couldn't find an answer. Can you rephrase your question?"


           logging.info(f"💬 LLM Fallback Response: {llm_fallback_answer}")

           return {"answer": llm_fallback_answer}

        logging.info(f"📌 Context sent to LLM: {context[:500]}...")

        # ✅ Step 4: Ask LLM to Generate Answer
        llm_prompt = f"""
        You are an AI tutor specializing in educational topics.
        Answer the following question based on the retrieved context.
        If no relevant data is found, generate an educational response.

        Question: {query}
        Context: {context if context else 'No relevant data available'}

        💡 **Rules:**
- If the question is about programming, provide a structured answer.
- If it's about AI, explain concepts in detail.
- If it's general knowledge, make it informative.
- If the query is non-educational, turn it into a learning opportunity.

🎯 **Your Task:** Provide the best, most educational response possible.
"""
#await
        llm_response =  chat(model="mistral", messages=[
            {"role": "system", "content": "You are an AI tutor. Always generate an answer, even when context is missing."},
            {"role": "user", "content": llm_prompt}
        ])

        # ✅ Extract LLM Response
        llm_answer = llm_response.get('message', {}).get('content', "").strip() if llm_response else "I couldn't find an answer. Can you rephrase your question?"

        if not llm_answer:  # Ensure the response isn't empty
           llm_answer = "I'm not sure, but let's explore this together! Try asking in a different way."


        # ✅ Step 5: Return Final Answer
        final_answer = f"{user_message}\n\n{llm_answer}" if user_message else llm_answer
        final_answer += "\n\n🤔 If you have more questions, feel free to ask!"
        return {"answer": final_answer or "I couldn't find an answer. Can you rephrase your question?"}
    
    
    

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


import httpx
import dateparser
import logging

async def get_on_this_day_events(query):
    """Fetch historical events for a given date using Wikipedia."""
    try:
        # Use dateparser to handle fuzzy dates like "aprl 6 2025"
        parsed_date = dateparser.parse(query, settings={"PREFER_DATES_FROM": "past"})
        if not parsed_date:
            return None

        month = parsed_date.month
        day = parsed_date.day

        url = f"https://en.wikipedia.org/api/rest_v1/feed/onthisday/events/{month}/{day}"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

        events = data.get("events", [])[:5]
        if not events:
            return None

        result = f"📅 **Historical Events on April {day}:**\n\n"
        for event in events:
            year = event.get("year")
            text = event.get("text")
            result += f"🔹 {year}: {text}\n"

        return result.strip()

    except Exception as e:
        logging.error(f"❌ Error fetching on-this-day history: {e}")
        return None


from dateutil import parser as date_parser
import re

def extract_date_from_query(query: str):
    try:
        match = re.search(r'(?:on\s+)?([A-Za-z]+\s+\d{1,2},\s*\d{4})', query)
        if match:
            return date_parser.parse(match.group(1), fuzzy=True)
    except Exception:
        return None



from dateutil import parser
from datetime import datetime

def is_future_date(text):
    try:
        parsed_date = parser.parse(text, fuzzy=True)
        return parsed_date > datetime.now()
    except:
        return False


@app.get("/query")
async def get_answer(query: str, chat_id: str = Query(default_factory=lambda: str(uuid4()))):

    """Handles user queries and retrieves relevant information using LLM."""
    logging.info(f"Received query: {query}| Chat ID: {chat_id}")

    # Save the user message
    save_message(chat_id, "user", query)

    query = query.lower().strip()
    if query in COMMON_RESPONSES:
        return {"response": COMMON_RESPONSES[query], "chat_id": chat_id}

    if len(query) < 3:  
        return {"response": "Could you provide more details?"}

    # 🔸 Collect context from all sources
    context_parts = []

    # 🔹 1. ChromaDB Retrieval
    try:
        chroma_result = await retrieve_relevant_text(query, model)
        if chroma_result:
            context_parts.append(chroma_result)
    except Exception as e:
        logging.error(f"❌ Error retrieving ChromaDB result: {e}", exc_info=True)

    # 🔹 2. Wikipedia
    try:
        wiki_result = await async_wikipedia_search(query)
        if wiki_result:
            logging.info(f"📝 Storing Wikipedia Data: {wiki_result}")
            store_text_in_chroma(wiki_result, "Wikipedia", model)
            context_parts.append(wiki_result)
        else:
            logging.info(f"⚠️ No relevant Wikipedia result found for '{query}'")
    except Exception as e:
        logging.error(f"❌ Error retrieving Wikipedia data: {e}")

    # 🔹 3. Google Search
    try:
        google_result = await search_google(query, model)
        if google_result:
            context_parts.extend(google_result)
    except Exception as e:
        logging.error(f"Error in retrieving Google results: {e}")

    # 🔹 4. Web Scrape & Summarize
    try:
        web_scrape_result = await scrape_and_summarize(query)
        if web_scrape_result:
            context_parts.append(web_scrape_result)
    except Exception as e:
        logging.error(f"Error in scraping and summarizing: {e}")

    # 🔹 4.5 Date-based historical news fetch
    try:
       extracted_date = extract_date_from_query(query)
       if extracted_date and extracted_date.date() < datetime.now().date():
          news_results = await fetch_latest_news("education", date=extracted_date)
          if news_results:
            context_parts.append("🗞️ News Highlights:\n" + "\n".join(news_results))
    except Exception as e:
        logging.error(f"❌ Error fetching historical news: {e}")


      # 🔹 5. News API
    try:
        if any(kw in query.lower() for kw in ["what happened", "news", "happened on", "latest events", "headlines"]) and not is_future_date(query):
            news_result = await fetch_latest_news(query)
            if news_result:
                formatted_news = "\n".join(news_result)
                logging.info("📰 Injecting NewsAPI result into context.")
                context_parts.append(f"Live News Headlines:\n{formatted_news}")
        elif is_future_date(query):
            context_parts.append("The date mentioned is in the future. I can't fetch future news, but I can help you explore historical events or predictions!")
    except Exception as e:
        logging.error(f"Error fetching news data: {e}")

    # 🔹 Wikipedia "On this day"
    try:
        if "what happened on" in query.lower() or "on this day" in query.lower():
           history_result = await get_on_this_day_events(query)
           if history_result:
              logging.info("📅 Injecting Wikipedia historical events into context.")
              context_parts.append(history_result)
    except Exception as e:
        logging.error(f"❌ Error fetching historical events: {e}")


    


    # 🔸 Combine all context
    combined_context = "\n\n".join(context_parts) if context_parts else "No external context available."

    # # 🔹 Final: LLM Generation
    # try:
    #     final_prompt =(
    #         f"Use the following context to answer the question as accurately as possible. "
    #         f"If the context does not help, answer from your own knowledge.\n\n"
    #         f"Context:\n{combined_context}\n\n"
    #         f"Question: {query}\nAnswer:"
    #     )

    #     final_answer = await generate_llm_response(final_prompt)

    #     return {"response": f"{final_answer}"}
    
    # 🔹 Final: LLM Generation
    
    final_prompt = f"""
    You are **Smart Sage**, a Retrieval-Augmented Generation (RAG) AI-powered educational assistant.  
Your mission is to help students learn, understand, and explore various topics — from programming to AI, general knowledge, and beyond. You always aim to educate, even when the input isn't directly academic.

---

📚 **Guidelines**:

1. If **relevant educational context** is available, use it to craft a clear and helpful response.
2. If the **query is non-educational**, creatively turn it into a **learning opportunity**.
   - Example: If asked about a celebrity, explain their cultural or societal impact.
   - If it's a fun fact or joke, explore the science or psychology behind it.
3. If it's a **completely unrelated or inappropriate query** (e.g., gossip, dating, etc.), respond politely:
   > "Hey! I'm Smart Sage, your educational companion. I focus on learning and knowledge-building. Could you please ask something study-related? 😊"

4. If the query is **programming-related**, respond with structured code, explanations, and comments.
5. If it's about **AI or technology**, provide deep and insightful information with relevant examples.
6. If it's a **general knowledge** question, keep the answer concise and meaningful.
7. Never say “I don’t know.” Always attempt to give a useful answer based on context or your own understanding.
8. Maintain a **friendly, clear, and student-first tone** — like a smart study buddy!
9. If the user asks about news, respond with the **latest educational news headlines** or summaries.
10. If there's a scheduled live session or reminder, **politely mention the date and time** in the reply.

---

    📚 **Context**:
    {combined_context if combined_context else "No relevant data available"}

    🧠 **Question**:
    {query}

    🎯 **Your Task**: Provide a friendly, helpful, and educational response like a human tutor.
    """
        
    try:
        # Build LLM history
        chat_history = get_chat_history(chat_id)
        chat_history.append({
            "role": "system",
            "content": "You are Smart Sage, an educational assistant using the following context."
        })
        chat_history.append({
            "role": "user",
            "content": final_prompt
        })


        llm_response = chat(model="mistral", messages=[
          {"role": "system", "content": "You are Smart Sage, a helpful educational AI assistant. Answer accurately using the context provided. If no context is available, use your own knowledge. If asked about recent topics, respond with up-to-date information."},
          {"role": "user", "content": final_prompt}
        ])

        final_answer = llm_response.get('message', {}).get('content', "").strip() if llm_response else "I couldn't find an answer. Can you rephrase your question?"

        # Save assistant reply
        save_message(chat_id, "assistant", final_answer)

        return {"response": final_answer, "chat_id": chat_id}

    except Exception as e:
      logging.error(f"❌ Error generating final LLM response: {e}", exc_info=True)
      return {"response": "I couldn't generate a response at the moment. Please try again later."}

    



# ✅ Initialize ChromaDB collection
chroma_client = PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection(
    name="study_materials",
    metadata={
        "hnsw:space": "cosine", 
        "hnsw:M": 128,
        "hnsw:ef_construction": 400,
        "hnsw:ef": 256
    }
)

from fastapi.responses import JSONResponse


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handles file uploads and stores the extracted content in ChromaDB."""
    try:
        # ✅ Ensure chroma_collection is passed
        response_message = await process_uploaded_file(file, model, chroma_collection)
        logging.info(f"✅ File '{file.filename}' processed and stored successfully.")
        return JSONResponse(content={"message": response_message})
    except Exception as e:
        logging.error(f"❌ Error processing file upload: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the file.")


import hashlib

def generate_hashed_id(text):
    """Generate a unique hashed ID based on the input text."""
    return hashlib.sha256(text.encode()).hexdigest()

def id_exists(chroma_collection, unique_id):
    """Check if a given ID already exists in ChromaDB."""
    results = chroma_collection.get(ids=[unique_id])
    return len(results['ids']) > 0

def add_embedding(chroma_collection, text, embedding, metadata):
    """Add a new embedding only if it does not already exist."""
    unique_id = generate_hashed_id(text)  # Generate a unique ID
    if not id_exists(chroma_collection, unique_id):
        chroma_collection.add(ids=[unique_id], embeddings=[embedding], metadatas=[metadata])
        print(f"Added: {unique_id}")
    else:
        print(f"Skipped (Duplicate): {unique_id}")

def remove_duplicate_ids(chroma_collection):
    """Remove duplicate embeddings based on their unique IDs."""
    existing_data = chroma_collection.get()
    seen_ids = set()
    for doc_id in existing_data['ids']:
        if doc_id in seen_ids:
            chroma_collection.delete(ids=[doc_id])  # Remove duplicate
            print(f"Removed duplicate: {doc_id}")
        else:
            seen_ids.add(doc_id)


async def generate_llm_response(query: str, context: str = None) -> str:
    llm_prompt = f"""
    You are **Smart Sage**, an AI-powered educational assistant created by Ragulan S. 🎓  
Your goal is to help students learn, understand, and explore various topics — from programming to AI, general knowledge, and more. You always aim to educate, even if the input isn't directly academic.

---

📚 **Instructions:**

If a user asks "Who are you?" or something similar, reply politely:
  > "I'm Smart Sage, your AI-powered study assistant created by Ragulan S to help you learn and grow! and more relatively "

1. If **relevant educational context** is available, use it to craft your response.
2. If the **query is non-educational**, creatively turn it into a **learning opportunity**.
   - E.g., If asked about a celebrity, explain their influence on society, media, or culture.
   - If it's a joke or fun fact, explain the underlying concept behind it (e.g., science of humor, psychology).
3. If it's a **completely unrelated or inappropriate query** (e.g., gossip, dating, etc.), respond **politely**:
   > "Hey! I'm Smart Sage, your educational companion. I focus on learning and knowledge-building. Could you please ask something study-related? 😊"

4. If the query is **programming-related**, respond with structured code, explanation, and comments.
5. If it's about **AI/tech**, go deep with clear explanations and examples.
6. If it's about **general knowledge**, make it concise yet insightful.
7. Never say “I don’t know.” Do your best to give a meaningful answer.
8. Use a **friendly, clear, and student-first tone** — like a smart study buddy!

---

    Question: {query}
    Context: {context if context else 'No relevant data available'}
    
🎓 **Smart Sage's Reply:**
"""

    try:
        response = await asyncio.to_thread(ollama.chat, model="mistral", messages=[
            {"role": "system", "content": "You are Smart Sage, a helpful educational AI assistant. Answer accurately using the context provided. If no context is available, use your own knowledge. If asked about recent topics, respond with up-to-date information."},
            {"role": "user", "content": llm_prompt}
        ])
        
        if response and 'message' in response:
            return response['message'].get('content', "I couldn't generate an answer. Try again.")
        else:
            return "⚠️ No response received from LLM."

    except Exception as e:
        logging.error(f"❌ LLM API Error: {e}")
        return "⚠️ There was an error processing your request. Please try again."





def get_response(query):
    collected_data = []

    # Step 1: Retrieve from ChromaDB
    chroma_result = chroma_collection.query(query)
    if chroma_result and chroma_result['documents']:
        collected_data.append("📚 ChromaDB: " + chroma_result['documents'][0])

    # Step 2: Retrieve from Google Search
    google_result = search_google(query, model)
    if google_result:
        collected_data.append("🌍 Google Search: " + google_result)

    # Step 3: Retrieve from Wikipedia API
    wiki_result = async_wikipedia_search(query)
    if wiki_result:
        collected_data.append("📖 Wikipedia: " + wiki_result)

    # Step 4: Collect All Sources (Ensures Data is Used)
    final_prompt = f"""
    You are an **AI tutor** that provides the most **accurate, educational, and structured** answers.

    **User Query:** "{query}"

    **Collected Data:**
    {chr(10).join(collected_data) if collected_data else "⚠️ No relevant data found from external sources."}

    🛠 **Your Task:**
    1️⃣ Analyze the given data and generate a clear, structured, and **educational** response.  
    2️⃣ If the collected data is **insufficient**, rely on your knowledge to provide an **accurate answer**.  
    3️⃣ Ensure responses are **detailed and informative** rather than short one-liners.  
    4️⃣ If the query is **non-educational**, convert it into a learning experience.

        You are **Smart Sage**, an AI-powered educational assistant . 
Your goal is to help students learn, understand, and explore various topics — from programming to AI, general knowledge, and more. You always aim to educate, even if the input isn't directly academic.

    **Final Answer:**  
    """

    # Step 5: Force LLM to Generate an Answer
    llm_response = generate_llm_response(final_prompt)
    
    return llm_response

# ---

# 📚 **Instructions:**

# 1. If **relevant educational context** is available, use it to craft your response.
# 2. If the **query is non-educational**, creatively turn it into a **learning opportunity**.
#    - E.g., If asked about a celebrity, explain their influence on society, media, or culture.
#    - If it's a joke or fun fact, explain the underlying concept behind it (e.g., science of humor, psychology).
# 3. If it's a **completely unrelated or inappropriate query** (e.g., gossip, dating, etc.), respond **politely**:
#    > "Hey! I'm Smart Sage, your educational companion. I focus on learning and knowledge-building. Could you please ask something study-related? 😊"

# 4. If the query is **programming-related**, respond with structured code, explanation, and comments.
# 5. If it's about **AI/tech**, go deep with clear explanations and examples.
# 6. If it's about **general knowledge**, make it concise yet insightful.
# 7. Never say “I don’t know.” Do your best to give a meaningful answer.
# 8. Use a **friendly, clear, and student-first tone** — like a smart study buddy!

# ---





if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
