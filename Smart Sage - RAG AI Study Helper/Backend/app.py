

# from fastapi import FastAPI, UploadFile, File, HTTPException
# from pydantic import BaseModel
# import uvicorn
# import ollama
# from Backend.retrieval import retrieve_relevant_text, process_uploaded_file, search_web
# from Backend.storage import store_text_in_chroma, get_chroma_collection
# import logging
# import sympy
# import re
# from sentence_transformers import SentenceTransformer 

# logging.basicConfig(level=logging.DEBUG)

# app = FastAPI()

# # Load the SentenceTransformer model
# model = SentenceTransformer('all-MiniLM-L6-v2') #add this line.
# logging.info("Model loaded successfully!") #add this line.


# def is_educational_query(query: str) -> bool:
#     """Determines if the query is educational using LLM classification and keyword-based backup."""
    
#     educational_keywords = {
#         "math", "science", "physics", "chemistry", "biology", "engineering", "computer science",
#         "programming", "history", "literature", "education", "algorithm", "data structure", 
#         "artificial intelligence", "machine learning", "deep learning", "statistics", "probability", 
#         "calculus", "algebra", "coding", "retrieval-augmented generation", "rag", "natural language processing", 
#         "neural networks", "transformer models", "data mining", "data science", "nlp", "chatbots"
#     }

#     query_lower = re.sub(r'[^\w\s]', '', query.lower()) #remove punctuation

#     if any(keyword in query_lower for keyword in educational_keywords):
#         logging.info(f"✅ Query classified as educational based on keywords: '{query}'")
#         return True

#     try:
#         response = ollama.chat(model="mistral", messages=[
#             {"role": "system", "content": "Classify whether the following query is educational "
#                                           "(math, science, engineering, history, literature, artificial intelligence, "
#                                           "machine learning, data science, programming, etc.). "
#                                           "Reply ONLY with 'yes' or 'no'—nothing else."},
#             {"role": "user", "content": query}
#         ])

#         classification = response.get('message', {}).get('content', "").strip().lower()

#         logging.info(f"🔍 LLM Classification Response: '{classification}' for Query: '{query}'")

#         if classification == "yes":
#             return True
#         elif classification == "no":
#             return False

#         logging.warning(f"⚠️ Unexpected LLM response: {classification}")
#         return False

#     except Exception as e:
#         logging.error(f"❌ LLM classification error: {e}", exc_info=True)
#         return False

# class QueryRequest(BaseModel):
#     question: str


# @app.post("/ask")
# async def answer_question(request: QueryRequest):
#     """Handles user queries, prioritizing mathematical evaluation and ChromaDB."""

#     try:
#         question = request.question.strip().lower()

#         # Attempt LLM-assisted math evaluation
#         try:
#             llm_response = ollama.chat(model="mistral", messages=[
#                 {"role": "system", "content": "Extract the mathematical expression from the following text and convert it to SymPy syntax. If no mathematical expression is found, simply return the original text."},
#                 {"role": "user", "content": question}
#             ])
#             processed_question = llm_response.get('message', {}).get('content', '')
#             logging.debug(f"LLM processed math question: {processed_question}") #log the LLM output.

#             result = sympy.sympify(processed_question)
#             if isinstance(result, sympy.Expr):
#                 answer = str(result)
#             else:
#                 answer = str(result)
#             logging.debug(f"Sympy answer: {answer}") #log the sympy answer.
#             return {"answer": answer}
#         except (sympy.SympifyError, TypeError, ValueError) as e:
#             logging.debug(f"SymPy error: {e}")
#             pass
#         except Exception as e:
#             logging.debug(f"LLM math process error: {e}")
#             pass

#         if not is_educational_query(request.question):
#             logging.debug("Question was not educational.")
#             return {"answer": "I'm here to assist with educational topics. Please ask about math, science, programming, or related subjects."}

#         context = retrieve_relevant_text(request.question, model) 
#         if context and "No relevant data found" not in context:
#             logging.info("ChromaDB retrieval used.")
#             return {"answer": context}

#         response = ollama.chat(model="mistral", messages=[
#     {
#         "role": "system",
#         "content": """
#         You are a highly skilled AI tutor specializing in educational topics, including math, science, programming, history, and literature. 
#         Your goal is to provide clear, concise, and accurate explanations and solutions.
#         When answering questions:
#         - Break down complex concepts into simpler terms.
#         - Provide step-by-step solutions for problems.
#         - Use examples and analogies to illustrate concepts.
#         - Encourage critical thinking by asking follow-up questions.
#         - If you don't understand a question, ask for clarification.
#         - If you are unsure of the answer, state that you are unsure.
#         """
#     },
#     {"role": "user", "content": request.question}
# ])
#         logging.info("LLM used for response.")
#         print(f"🔍 Ollama Response: {response}")

#         llm_answer = response.get('message', {}).get('content', "")

#         if llm_answer and "I couldn't find a relevant answer." not in llm_answer:
#             return {"answer": llm_answer}

#         try:
#             search_results = search_web(request.question + " latest research site:nature.com OR site:sciencedirect.com OR site:nasa.gov")
#             if search_results and search_results[0] != "No relevant results found.":
#                 logging.info("Web search used.")
#                 #summarize results.
#                 response = ollama.chat(model = "mistral", messages = [{"role": "system", "content": "Summarize the following text."}, {"role": "user", "content": f"{search_results}"}])
#                 return {"answer": f"🔎 Here's the latest research update:\n{response.get('message', {}).get('content', '')}"}
#         except Exception as e:
#             logging.error(f"Web search error: {e}")

#         return {"answer": "I couldn't find a relevant answer from any sources. Try rephrasing your question."}

#     except Exception as e:
#         logging.error(f"Error processing request: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again later.")


# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     """Handles file uploads for study materials (PDF, DOCX, TXT)."""
#     try:
#         text = process_uploaded_file(file, model)
#         print(f"Extracted Text: {text}")
#         store_text_in_chroma(text, file.filename)
#         logging.info(f"File '{file.filename}' processed and stored successfully.")
#         return {"message": "File processed and stored successfully"}
#     except ValueError as ve:
#         raise HTTPException(status_code=400, detail=str(ve))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, UploadFile, File, HTTPException,Form
from pydantic import BaseModel
import uvicorn
import ollama
from Backend.retrieval import retrieve_relevant_text, process_uploaded_file, search_web,scrape_educational_websites
from Backend.storage import store_text_in_chroma, get_chroma_collection
import logging
import sympy
import re
from sentence_transformers import SentenceTransformer
import asyncio
import httpx
import wikipedia

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')
logging.info("Model loaded successfully!")

def is_educational_query(query: str) -> bool:
    # ... (Your existing is_educational_query function)
    return True #replace with your existing function.

class QueryRequest(BaseModel):
    question: str

async def async_web_search(question):
    try:
        async with httpx.AsyncClient() as client:
            url = f"https://api.duckduckgo.com/?q={question}&format=json"
            response = await client.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            results = [topic["Text"] for topic in data.get("RelatedTopics", []) if "Text" in topic]
            return results if results else ["No relevant results found."]
    except httpx.RequestError as e:
        logging.error(f"Async web search failed: {e}")
        return ["Web search error."]
    except Exception as e:
        logging.error(f"Async unexpected web search error: {e}")
        return ["Unexpected error occurred during web search."]

# Load ChromaDB collection once at startup
chroma_collection = get_chroma_collection()

async def async_chroma_retrieval(question, model):
    """Retrieve stored embeddings from ChromaDB without repeated fetching."""
    try:
        return retrieve_relevant_text(question, model, chroma_collection)  # Pass preloaded collection
    except Exception as e:
        logging.error(f"Async ChromaDB retrieval error: {e}")
        return "ChromaDB retrieval failed."

@app.post("/ask")
async def answer_question(request: QueryRequest):
    """Handles user queries with LLM as the central analysis engine, asynchronously."""
    try:
        question = request.question.strip().lower()

        # 1. ChromaDB Retrieval and 2. Web Search (Concurrent)
        web_search_task = asyncio.create_task(async_web_search(question + " latest research site:nature.com OR site:sciencedirect.com OR site:nasa.gov"))
        chroma_task = asyncio.create_task(async_chroma_retrieval(question, model))

        web_results = await web_search_task
        context = await chroma_task

        # 3. LLM Analysis and Response
        try:
            llm_prompt = f"""
            You are a highly skilled AI tutor specializing in educational topics.
            Analyze the following question and provide a comprehensive answer using the provided context and web search results.

            Question: {question}

            ChromaDB Context: {context}

            Web Search Results: {web_results}

            Provide a well-structured and detailed response, including:
            - Clear explanations of concepts.
            - Step-by-step solutions for problems.
            - Summaries of relevant research.
            - Use examples and analogies.
            - If you don't find the answer, please say that you did not find the answer.
            """

            llm_response = ollama.chat(model="mistral", messages=[
                {"role": "system", "content": "You are an expert AI tutor."},
                {"role": "user", "content": llm_prompt}
            ])
            llm_answer = llm_response.get('message', {}).get('content', "")

            try:
                result = sympy.sympify(llm_answer)
                if isinstance(result, sympy.Expr):
                    answer = str(result)
                else:
                    answer = str(result)
                return {"answer": answer}
            except:
                pass

            if llm_answer and "I couldn't find a relevant answer." not in llm_answer:
                logging.info("LLM used for response.")
                return {"answer": llm_answer}

        except Exception as e:
            logging.error(f"LLM processing error: {e}")

        # 4. Wikipedia (Fallback)
        try:
            search_results = wikipedia.search(request.question)
            if search_results:
                wiki_summary = wikipedia.summary(search_results[0])
                logging.info(f"✅ Wikipedia Result: {wiki_summary}")
                return f"Wikipedia says: {wiki_summary}"
        except wikipedia.exceptions.DisambiguationError as e:
            logging.error(f"❌ Wikipedia DisambiguationError: {e}")
            return f"Wikipedia found multiple meanings for '{request.question}': {e}"
        except wikipedia.exceptions.PageError as e:
            logging.error(f"❌ Wikipedia PageError: {e}")
            return "Wikipedia page not found."
        except Exception as e:
            logging.error(f"Wikipedia error: {e}")

        return {"answer": "I couldn't find a relevant answer from any sources. Try rephrasing your question."}

    except Exception as e:
        logging.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    

# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     """Handles file uploads for study materials (PDF, DOCX, TXT)."""
#     try:
#         text = process_uploaded_file(file, model)
#         print(f"Extracted Text: {text}")
#         store_text_in_chroma(text, file.filename)
#         logging.info(f"File '{file.filename}' processed and stored successfully.")
#         return {"message": "File processed and stored successfully"}
#     except ValueError as ve:
#         raise HTTPException(status_code=400, detail=str(ve))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    """Handles file uploads and stores the extracted content in ChromaDB."""
    logging.info(f"Processing uploaded file: {file.filename}")
    try:
        result = process_uploaded_file(file, model)
        return {"message": result}
    except Exception as e:
        logging.error(f"Error processing file upload: {e}")
        return {"message": "An error occurred while processing the file."}
    


async def async_wikipedia_search(query):
    """Asynchronously fetch Wikipedia summary."""
    try:
        search_results = wikipedia.search(query)
        if search_results:
            return wikipedia.summary(search_results[0], sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Wikipedia found multiple meanings for '{query}': {e}"
    except wikipedia.exceptions.PageError:
        return "Wikipedia page not found."
    except Exception as e:
        logging.error(f"Wikipedia search error: {e}")
    return None

@app.get("/query")
async def get_answer(query: str):
    """Handles user queries by searching ChromaDB, Wikipedia, and Educational Websites asynchronously."""
    logging.info(f"Received query: {query}")

    async def fetch_chroma():
        try:
            result = retrieve_relevant_text(query, model)
            return result if "No relevant data found" not in result else None
        except Exception as e:
            logging.error(f"Error retrieving data from ChromaDB: {e}")
            return None

    async def fetch_edu_results():
        try:
            return scrape_educational_websites(query)
        except Exception as e:
            logging.error(f"Error scraping educational websites: {e}")
            return None

    async def fetch_web_search():
        try:
            return search_web(query)
        except Exception as e:
            logging.error(f"Error performing web search: {e}")
            return None

    chroma_task = asyncio.create_task(fetch_chroma())
    edu_task = asyncio.create_task(fetch_edu_results())
    wiki_task = asyncio.create_task(async_wikipedia_search(query))
    web_task = asyncio.create_task(fetch_web_search())

    chroma_result, edu_result, wiki_result, web_result = await asyncio.gather(
        chroma_task, edu_task, wiki_task, web_task
    )

    if chroma_result:
        return {"response": chroma_result}
    if edu_result:
        return {"response": "\n".join(edu_result)}
    if wiki_result:
        return {"response": wiki_result}
    if web_result:
        return {"response": "\n".join(web_result)}

    return {"response": "No relevant answer found."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)