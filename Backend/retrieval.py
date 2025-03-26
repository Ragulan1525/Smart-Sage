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
import httpx
from transformers import pipeline

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

logging.basicConfig(level=logging.DEBUG)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
logging.info("✅ Model loaded successfully!")

# Load ChromaDB collection once
chroma_collection = get_chroma_collection()

# Educational websites for scraping
EDUCATIONAL_SITES = [
    "https://www.khanacademy.org/",
    "https://www.coursera.org/",
    "https://www.edx.org/",
    "https://www.geeksforgeeks.org/"
]


async def search_google(query):
    """Fetch live search results and extract summarized content."""
    try:
        async with httpx.AsyncClient() as client:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "q": query,
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_SEARCH_ENGINE_ID,
                "num": 3,  
            }
            response = await client.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data.get("items"):
                logging.warning(f"⚠️ No search results for: {query}")
                return None  

            extracted_results = []
            for item in data["items"]:
                page_title = item["title"]
                page_link = item["link"]
                page_text = await scrape_and_summarize(page_link)  # Get summarized text

                if page_text:
                    extracted_results.append(f"**{page_title}**\n{page_text}\n🔗 {page_link}")

            return extracted_results if extracted_results else None
    except Exception as e:
        logging.error(f"❌ Google Search API Error: {e}")
        return None



async def fetch_latest_news(query):
    """Fetches live news related to the query from NewsAPI asynchronously."""
    try:
        if not NEWS_API_KEY:
            raise ValueError("❌ Missing NewsAPI Key! Check your .env file.")

        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en&pageSize=5"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

        return [
            f"{article['title']}: {article['url']}"
            for article in data.get("articles", [])
        ] if data.get("articles") else None
    except Exception as e:
        logging.error(f"❌ News API Error: {e}")
        return None



# ✅ Load Summarization Model (One-time)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

async def scrape_and_summarize(url):
    """Scrapes a webpage and summarizes its content."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # ✅ Extract text from meaningful content tags
        content_elements = soup.find_all(['p', 'div', 'article', 'section'])
        text_parts = [element.get_text(separator=" ", strip=True) for element in content_elements]
        text = " ".join(text_parts)

        logging.debug(f"Scraped Text from {url}: {text[:300]}...")  # Log first 300 characters

        if len(text) < 100:
            return None  # Ignore irrelevant pages
        elif len(text) <= 500:
            return text  # Return short text as-is

        # ✅ Summarize if text is too long
        summary = summarizer(text[:1024], max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        logging.debug(f"Summarized Text from {url}: {summary}")
        return summary

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
    except Exception as e:
        logging.error(f"Unexpected web search error: {e}")
        return ["Unexpected error occurred during web search."]

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


async def async_wikipedia_search(query):
    """Asynchronously fetch Wikipedia summary."""
    try:
        search_results = await asyncio.to_thread(wikipedia.search, query)
        if not search_results:
            return None  

        try:
            summary = await asyncio.to_thread(wikipedia.summary, search_results[0], sentences=2, auto_suggest=False)
            return summary if len(summary) > 50 else None  

        except wikipedia.exceptions.DisambiguationError as e:
            if e.options:
                try:
                    summary = await asyncio.to_thread(wikipedia.summary, e.options[0], sentences=2, auto_suggest=False)
                    return summary if len(summary) > 50 else None  
                except Exception:
                    return None
            else:
                return None

    except Exception as e:
        logging.error(f"❌ Wikipedia Error: {e}")
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

        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=num_results
        )

        if not results or not results.get("documents") or len(results["documents"]) == 0:
            logging.warning("⚠️ No relevant documents found in ChromaDB.")
            return None

        retrieved_docs = results.get("documents", [])[0]

        # ✅ Store new data in ChromaDB if missing
        if not retrieved_docs or len(retrieved_docs[0]) == 0:
            logging.info("❌ No relevant data in ChromaDB. Fetching live results...")
            web_results = await search_google(query)

            if web_results:
                new_data = "\n".join(web_results)
                store_text_in_chroma(new_data, f"Google Data ({query})", model)
                return new_data

            return None  

        # ✅ Return first valid result
        return retrieved_docs[0][0]

    except Exception as e:
        logging.error(f"❌ ChromaDB retrieval error: {e}")
        return None


def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    try:
        reader = PyPDF2.PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip() or "No readable text found in the PDF."
    except Exception as e:
        logging.error(f"PDF read error: {e}")
        return "Error reading PDF file."

def extract_text_from_docx(file):
    """Extracts text from a DOCX file."""
    try:
        doc = docx.Document(file)
        return "\n".join(para.text.strip() for para in doc.paragraphs if para.text.strip()) or "No readable text found in the DOCX file."
    except Exception as e:
        logging.error(f"DOCX file read error: {e}")
        return "Error reading DOCX file."
    

def store_text_in_chroma(text, source_name, model):
    """Stores extracted text into ChromaDB, preventing duplicates."""
    try:
        existing_docs = chroma_collection.get(include=["documents"]).get("documents", [])
        
        if text in existing_docs:
            logging.info(f"🔄 Skipping duplicate entry from {source_name}")
            return
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        embeddings = [model.encode(chunk).tolist() for chunk in chunks]

        existing_docs = chroma_collection.get(include=["documents"]).get("documents", [])

        unique_chunks = [chunk for chunk in chunks if chunk not in existing_docs]
        if not unique_chunks:
          logging.info("🔄 No new data to add to ChromaDB.")

        chroma_collection.add(
           documents=unique_chunks,
           embeddings=[model.encode(chunk).tolist() for chunk in unique_chunks],
           ids=[str(hash(chunk)) for chunk in unique_chunks]
)

        logging.info(f"✅ Stored {len(chunks)} new chunks from {source_name} in ChromaDB!")

    except Exception as e:
        logging.error(f"❌ Error storing {source_name} data in ChromaDB: {e}")


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

def process_uploaded_file(file, model, chroma_collection):
    """Processes uploaded files and stores embeddings in ChromaDB."""
    try:
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file.file)
        elif file.filename.endswith(".docx"):
            text = extract_text_from_docx(file.file)
        else:
            return "Unsupported file format. Please upload a PDF or DOCX file."

        if text:
            store_text_in_chroma(text, model, chroma_collection)
            return "File processed and stored successfully."
        return "No valid text found in the uploaded file."
    except Exception as e:
        logging.error(f"Error processing uploaded file: {e}")
        return "Error processing uploaded file."