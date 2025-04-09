import chromadb
from sentence_transformers import SentenceTransformer
import wikipedia
import requests
import os
import logging
import asyncio
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from Backend.storage import get_chroma_collection
from langchain.text_splitter import RecursiveCharacterTextSplitter
from newspaper import Article
import PyPDF2
import docx
from docx import Document
import fitz
import httpx
import uuid
import validators
import re
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from pdfminer.high_level import extract_text as extract_text_from_pdf

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
DUCKDUCKGO_API_URL = "https://api.duckduckgo.com/"
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

logging.basicConfig(level=logging.DEBUG)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
logging.info("✅ Model loaded successfully!")

# Load ChromaDB collection once
chroma_collection = get_chroma_collection()


# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="study_docs")


nltk.download("punkt")

# Educational websites for scraping
EDUCATIONAL_SITES = [
    "https://www.khanacademy.org/",
    "https://www.coursera.org/",
    "https://www.edx.org/",
    "https://www.geeksforgeeks.org/"
]


# ✅ Load Summarization Model (One-time)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


async def search_google(query, model, retries=3):
    """Fetch search results from Google and fallback to DuckDuckGo on error."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_SEARCH_ENGINE_ID,
        "num": 5,
        "hl": "en"
    }

    for attempt in range(retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if not data.get("items"):
                    logging.warning(f"⚠️ No results from Google. Switching to DuckDuckGo.")
                    return await search_duckduckgo(query, model)

                extracted_results = []
                for item in data["items"]:
                    title, snippet, link = item.get("title", ""), item.get("snippet", ""), item.get("link", "")

                    if not link or not validators.url(link) or any(x in link for x in ["youtube.com", "reddit.com"]):
                        continue

                    page_text = await scrape_and_summarize(link)
                    result_entry = f"🔹 **{title}**\n{snippet}\n🔗 {link}"
                    if page_text:
                        result_entry += f"\n📄 Summary: {page_text}"

                    extracted_results.append(result_entry)

                if extracted_results:
                    await store_text_in_chroma("\n".join(extracted_results), f"Google Data ({query})", model)

                return extracted_results

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait_time = 2 ** attempt
                logging.warning(f"🚨 Rate limit hit! Waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logging.error(f"❌ Google API error: {e}")
                break

        except Exception as e:
            logging.error(f"❌ Unexpected error: {e}")
            break
    
    if "ieeexplore.ieee.org" in url and response.status_code == 418:
       return {
        "type": "link-only",
        "message": (
            "🔒 The requested IEEE Xplore page cannot be accessed directly due to restrictions on automated scraping. "
            "You can view the document manually here:\n\n"
            f"{url}\n\n"
            "For metadata access, consider using the [IEEE Xplore API](https://developer.ieee.org/)."
        )
    }


    # Fallback if retries failed
    return await search_duckduckgo(query, model)


async def search_duckduckgo(query, model):
    """Fetches search results from DuckDuckGo if Google fails."""
    try:
        async with httpx.AsyncClient() as client:
            params = {"q": query, "format": "json"}
            response = await client.get(DUCKDUCKGO_API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if '50x.html' in str(response.url):
                logging.warning("⚠️ DuckDuckGo API returned a redirect. Possibly malformed input.")

            related_topics = data.get("RelatedTopics", [])
            extracted_results = []

            for topic in data.get("RelatedTopics", []):
              if "Text" in topic and "FirstURL" in topic:
                url = topic["FirstURL"]
                if not url.startswith("http"):
                  continue
                extracted_results.append(topic["Text"])

            if extracted_results:
                await store_text_in_chroma("\n".join(extracted_results), f"DuckDuckGo Data ({query})", model)

            if not extracted_results:
                logging.warning(f"⚠️ No useful DuckDuckGo results found for '{query}'")


            return extracted_results

    except Exception as e:
        logging.error(f"❌ DuckDuckGo API error: {e}")
        return None


async def fetch_latest_news(query):
    """Fetches the latest news from NewsAPI."""
    try:
        if not NEWS_API_KEY:
            raise ValueError("❌ Missing NewsAPI Key!")

        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en&pageSize=5"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

        return [f"{article['title']}: {article['url']}" for article in data.get("articles", [])] if data.get("articles") else None
    except Exception as e:
        logging.error(f"❌ News API Error: {e}")
        return None


async def scrape_and_summarize(url):
    """Scrapes a webpage and summarizes its content."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.extract()

        content_elements = soup.find_all(["p", "div", "article", "section"])
        text = " ".join([elem.get_text(separator=" ", strip=True) for elem in content_elements]).strip()

        if len(text) < 100:
            return None  
        elif len(text) <= 500:
            return text  

        text = text[:1024]
        try:
            summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
            return summary
        except Exception as e:
            logging.error(f"❌ Summarization error: {e}")
            return text[:300]

    except Exception as e:
        logging.error(f"❌ Error scraping {url}: {e}")
        return None
        
    
# ✅ Helper Function to Extract First Full Sentences (Avoid Mid-Cutoff)
def extract_first_sentences(text, max_length=500):
    """Extracts the first few full sentences without cutting off mid-way."""
    sentences = text.split(". ")
    output = ""
    for sentence in sentences:
        if len(output) + len(sentence) > max_length:
            break
        output += sentence + ". "
    return output.strip()


# def search_web(query):
#     """Performs a web search using DuckDuckGo API."""
#     try:
#         url = f"https://api.duckduckgo.com/?q={query}&format=json"
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         data = response.json()
#         results = [topic["Text"] for topic in data.get("RelatedTopics", []) if "Text" in topic]
#         return results if results else ["No relevant results found."]
#     except requests.exceptions.RequestException as e:
#         logging.error(f"Web search failed: {e}")
#         return ["Web search error."]
#     except Exception as e:
#         logging.error(f"Unexpected web search error: {e}")
#         return ["Unexpected error occurred during web search."]

import requests


def search_web(query):
    """Performs a web search using DuckDuckGo API."""
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = [topic["Text"] for topic in data.get("RelatedTopics", []) if "Text" in topic]
        return results if results else ["No relevant results found."]
    except requests.exceptions.RequestException as e:
        logging.error(f"Web search failed: {e}")
        return ["Web search error."]


async def scrape_educational_websites(query, model):
    """Scrapes educational websites for study material using semantic similarity."""
    results = []
    query_embedding = model.encode(query)

    for site in EDUCATIONAL_SITES:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(site, timeout=10)
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            content_elements = soup.find_all(['p', 'div', 'article', 'section', 'span'])
            text = " ".join([element.get_text(separator=" ", strip=True) for element in content_elements])

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = text_splitter.split_text(text)

            for chunk in chunks:
                chunk_embedding = model.encode(chunk)
                similarity = chunk_embedding @ query_embedding
                if similarity > 0.6:
                    results.append((similarity, f"From {site}: {chunk}"))

        except Exception as e:
            logging.error(f"Error fetching or processing {site}: {e}")

    results.sort(key=lambda x: x[0], reverse=True)
    return [result[1] for result in results[:3]]


import wikipedia

def async_wikipedia_search(query):
    """Fetch Wikipedia summary with proper error handling and debugging."""
    try:
        print(f"🔎 Searching Wikipedia for: {query}")
        search_results = wikipedia.search(query)

        print(f"📜 Wikipedia Search Results: {search_results}")

        if search_results:
           page = wikipedia.page(search_results[0])
           content = page.content
        else:
           logging.warning("⚠️ No relevant Wikipedia result found for '%s'", query)

        if not search_results:
            return None

        for result in search_results[:3]:
            try:
                page = wikipedia.page(result, auto_suggest=True)
                summary = wikipedia.summary(page.title, sentences=3)

                if query.lower() in page.title.lower() and len(summary) > 50:
                    return summary

            except wikipedia.exceptions.DisambiguationError as e:
                print(f"⚠️ Disambiguation Error: {e}")
                for option in e.options[:3]:
                    try:
                        summary = wikipedia.summary(option, sentences=3)
                        if len(summary) > 50:
                            return summary
                    except:
                        continue

            except wikipedia.exceptions.PageError:
                print(f"❌ PageError for {result}")
                continue

        return None

    except Exception as e:
        print(f"❌ Wikipedia API General Error: {e}")
        return None

    

async def async_scrape_web(query):
    """Asynchronously scrape educational websites for content if needed."""
    return scrape_educational_websites(query, model) or None



async def retrieve_relevant_text(query: str, model, top_k=3):
    """Retrieve relevant study material asynchronously from ChromaDB."""
    try:
        query_embedding = model.encode(query).tolist()
        available_docs = chroma_collection.count()
        num_results = max(1, min(top_k, available_docs))

        results = await asyncio.to_thread(
            chroma_collection.query, 
            query_embeddings=[query_embedding], 
            n_results=num_results
        )

        if not results or "documents" not in results or not results["documents"]:
            logging.warning("⚠️ No relevant documents found in ChromaDB.")
            return "No relevant results found."

        retrieved_docs = results["documents"][0] if results["documents"] else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        cleaned_results = []
        for idx, text in enumerate(retrieved_docs):
            if text and distances[idx] < 0.3:
                text = re.sub(r"[\n\t]+", " ", text)  # Remove line breaks/tabs
                text = re.sub(r"http\S+", "", text)  # Remove links
                text = re.sub(r"[^a-zA-Z0-9.,!?'\- ]+", "", text)  # ✅ Keeps hyphens and apostrophes
                text = re.sub(r"\s+", " ", text).strip()
                
                if len(text.split()) > 5:
                    cleaned_results.append(text)

        return "\n\n".join(cleaned_results) if cleaned_results else "No relevant results found."

    except Exception as e:
        logging.error(f"❌ ChromaDB retrieval error: {e}")
        return f"Error retrieving from ChromaDB: {str(e)}"
    


import fitz  # PyMuPDF

def extract_text_from_pdf(file_obj):
    """Attempts text extraction from PDF using PyMuPDF, with pdfminer fallback."""
    try:
        with fitz.open(stream=file_obj.file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        if not text.strip():
            raise ValueError("Empty text extracted with PyMuPDF.")
        return text
    except Exception as e:
        logging.warning(f"⚠️ PyMuPDF failed: {e}. Trying fallback with pdfminer.")
        # Rewind file pointer and use pdfminer
        file_obj.file.seek(0)
        return extract_text_from_pdf_using_pdfminer(file_obj.file)


    

from pdfminer.high_level import extract_text

def extract_text_from_pdf_using_pdfminer(file_path):
    """Extracts text using pdfminer.six (fallback method)."""
    try:
        return extract_text(file_path)
    except Exception as e:
        logging.error(f"pdfminer read error: {e}")
        return None

    

def extract_text_from_docx(file):
    """Extracts text from a DOCX file."""
    try:
        doc = docx.Document(file)
        extracted_text = "\n".join(para.text.strip() for para in doc.paragraphs if para.text.strip())
        return extracted_text if extracted_text else None  # ✅ Return None instead of a fixed message
    except Exception as e:
        logging.error(f"DOCX file read error: {e}")
        return None
    

import asyncio
import uuid
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

async def store_text_in_chroma(text, source_name, model, user_id=None):
    try:
        safe_source = re.sub(r'\W+', '_', source_name)

        existing_docs = chroma_collection.get(include=["documents", "metadatas"]) or {}
        existing_texts = existing_docs.get("documents", []) or []
        existing_metadata = existing_docs.get("metadatas", []) or []

        if user_id:
            for i, doc in enumerate(existing_texts):
                if doc == text and existing_metadata[i].get("user_id") == user_id:
                    logging.info(f"🔄 Skipping duplicate for user {user_id} from {source_name}")
                    return
        else:
            if text in existing_texts:
                logging.info(f"🔄 Skipping duplicate entry from {source_name}")
                return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_text(text)

        embeddings = await asyncio.to_thread(lambda: [model.encode(chunk).tolist() for chunk in chunks])
        ids = [f"{safe_source}_{uuid.uuid4()}" for _ in chunks]
        metadata = [{"source": source_name, "user_id": user_id} for _ in chunks]

        # Not awaited — ChromaDB add is usually synchronous
        chroma_collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata
        )

        logging.info(f"✅ Stored {len(chunks)} chunks from {source_name} into ChromaDB for user {user_id}.")

    except Exception as e:
        logging.error(f"❌ Error storing in ChromaDB: {e}")



def print_all_chroma():
    """Prints all documents stored in ChromaDB."""
    try:
        all_data = chroma_collection.get(include=["documents"])
        if all_data["documents"]:
            for doc in all_data["documents"]:
                print(f"Stored Document: {doc[:200]}...")  # Print first 200 chars
        else:
            print("⚠️ ChromaDB is empty!")
    except Exception as e:
        logging.error(f"Error retrieving ChromaDB data: {e}")



def clear_chroma_collection(chroma_collection):
    """Clears all data from the ChromaDB collection."""
    try:
        chroma_collection.delete(ids=chroma_collection.get()['ids'])
        logging.info("ChromaDB collection cleared.")
    except Exception as e:
        logging.error(f"Error clearing ChromaDB: {e}")

def print_all_chroma(chroma_collection):
    """Prints all documents in the ChromaDB collection."""
    try:
        all_data = chroma_collection.get(include=["documents"])
        if all_data["documents"]:
            for doc in all_data["documents"]:
                print(f"Document: {doc}")
        else:
            print("ChromaDB collection is empty.")
    except Exception as e:
        logging.error(f"Error printing ChromaDB data: {e}")

async def process_uploaded_file(file, model, chroma_collection):
    """Processes uploaded files and stores embeddings in ChromaDB."""
    try:
        logging.info(f"📁 Received file: {file.filename}")

        # Step 1: Extract text
        if file.filename.endswith(".pdf"):
            logging.info("📄 Detected PDF format.")
            text = extract_text_from_pdf(file)
        elif file.filename.endswith(".docx"):
            logging.info("📄 Detected DOCX format.")
            text = extract_text_from_docx(file)
        else:
            return "Unsupported file format. Please upload a PDF or DOCX file."

        logging.info(f"📝 Extracted text length: {len(text) if text else 0}")

        # Step 2: Store in Chroma
        if text:
            await store_text_in_chroma(text, file.filename, model, chroma_collection)
            return "File processed and stored successfully."
        
        return "No valid text found in the uploaded file."

    except Exception as e:
        logging.error(f"❌ Error processing uploaded file: {e}", exc_info=True)
        return "Error processing uploaded file."


