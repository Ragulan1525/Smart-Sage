# from fastapi import FastAPI, UploadFile, File, HTTPException
# from pydantic import BaseModel
# import uvicorn
# import ollama
# #import chroma_db
# import wikipediaapi
# from Backend.retrieval import retrieve_relevant_text, process_uploaded_file,search_web,scrape_educational_websites

# from Backend.storage import store_text_in_chroma, get_chroma_collection
# import logging

# logging.basicConfig(level=logging.DEBUG)

# # # Initialize Wikipedia API
# # wiki_wiki = wikipediaapi.Wikipedia('en')

# app = FastAPI()

# def is_educational_query(query: str) -> bool:
#     """Determines if the query is educational using LLM classification and keyword-based backup."""
    
#     educational_keywords = {
#         "math", "science", "physics", "chemistry", "biology", "engineering", "computer science",
#         "programming", "history", "literature", "education", "algorithm", "data structure", 
#         "artificial intelligence", "machine learning", "deep learning", "statistics", "probability", 
#         "calculus", "algebra", "coding", "retrieval-augmented generation", "rag", "natural language processing", 
#         "neural networks", "transformer models", "data mining", "data science", "nlp", "chatbots"
#     }

#     query_lower = query.lower()

#     # 🔹 1️⃣ Direct Keyword Match (to avoid unnecessary LLM calls)
#     if any(keyword in query_lower for keyword in educational_keywords):
#         logging.info(f"✅ Query classified as educational based on keywords: '{query}'")
#         return True  # If the query matches known educational terms, return True immediately

#     try:
#         # 🔹 2️⃣ LLM Classification (if keyword check fails)
#         response = ollama.chat(model="mistral", messages=[
#             {"role": "system", "content": "Classify whether the following query is educational "
#                                           "(math, science, engineering, history, literature, artificial intelligence, "
#                                           "machine learning, data science, programming, etc.). "
#                                           "Reply ONLY with 'yes' or 'no'—nothing else."},
#             {"role": "user", "content": query}
#         ])

#         classification = response.get('message', {}).get('content', "").strip().lower()

#         # ✅ Debug: Log what the LLM is returning
#         logging.info(f"🔍 LLM Classification Response: '{classification}' for Query: '{query}'")

#         # 🔹 3️⃣ Ensure clean response
#         if classification == "yes":
#             return True
#         elif classification == "no":
#             return False

#         # 🔹 4️⃣ Handle Unexpected LLM Responses (Failsafe)
#         logging.warning(f"⚠️ Unexpected LLM response: {classification}")
#         return False  # Default to False if response is unclear

#     except Exception as e:
#         logging.error(f"❌ LLM classification error: {e}", exc_info=True)
#         return False  # Default to False if LLM fails


# class QueryRequest(BaseModel):
#     question: str

# # @app.post("/ask")
# # async def answer_question(request: QueryRequest):
# #     """Handles user queries, filtering non-educational ones, and ensuring relevant responses."""
    
# #     if not is_educational_query(request.question):
# #         return {"answer": "I'm here to assist with educational topics. Please ask about math, science, programming, or related subjects."}

# #     try:
# #         context = retrieve_relevant_text(request.question)

# #         # If retrieved text is too large, summarize it
# #         if context and len(context) > 1000:  # Adjust threshold as needed
# #             response = ollama.chat(model="llama3", messages=[
# #                 {"role": "system", "content": "Summarize the following educational content to answer the user's question in a concise manner, the answer should be related to the user's query."},
# #                 {"role": "user", "content": f"Question: {request.question}\n\nContext: {context}"}
# #             ])
# #             return {"answer": response.get('message', {}).get('content', "I couldn't find a relevant answer.")}

# #         # If retrieval fails, use LLM directly
# #         if not context or "No relevant data found" in context:
# #             response = ollama.chat(model="llama3", messages=[
# #                 {"role": "system", "content": "You are an AI tutor. Answer educational questions based on your knowledge."},
# #                 {"role": "user", "content": request.question}
# #             ])
# #             return {"answer": response.get('message', {}).get('content', "I couldn't find a relevant answer.")}

# #         return {"answer": context}  # Return only the relevant extracted text

# #     except Exception as e:
# #         logging.error(f"Error processing request: {e}", exc_info=True)
# #         raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again later.")

# ############################################################

# # def search_wikipedia(query):
# #     """Search Wikipedia for relevant educational content."""
# #     page = wiki_wiki.page(query)
# #     return page.summary if page.exists() else None


# # #############################################################33
# # @app.post("/ask")
# # async def answer_question(request: QueryRequest):
# #     """Handles user queries with structured query resolution."""
    
# #     try:
# #         if not is_educational_query(request.question):
# #             return {"answer": "I'm here to assist with educational topics. Please ask about math, science, programming, or related subjects."}

# #         context = retrieve_relevant_text(request.question)

# #         if context and "No relevant data found" not in context:
# #             return {"answer": context}
        
# #         # Perform Web Search Safely
# #         try:
# #             search_results = search_web(request.question + " latest research site:nature.com OR site:sciencedirect.com OR site:nasa.gov")
# #             if search_results and search_results[0] != "No relevant results found.":
# #                 return {"answer": f"Here's the latest research update:\n{search_results[0]}"}
# #         except Exception as e:
# #             logging.error(f"Web search error: {e}")

# #         # Fallback to AI Tutor
# #         response = ollama.chat(model="mistral", messages=[
# #             {"role": "system", "content": "You are an AI tutor that provides educational responses."},
# #             {"role": "user", "content": request.question}
# #         ])

# #         return {"answer": response.get('message', {}).get('content', "I couldn't find a relevant answer.")}

# #     except Exception as e:
# #         logging.error(f"Error processing request: {e}", exc_info=True)
# #         raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again later.")




# @app.post("/ask")
# async def answer_question(request: QueryRequest):
#     """Handles user queries, prioritizing ChromaDB before using AI-generated responses or web search."""

#     try:
#         # ✅ Ensure only educational queries are handled
#         if not is_educational_query(request.question):
#             return {"answer": "I'm here to assist with educational topics. Please ask about math, science, programming, or related subjects."}

#         # 🔹 1️⃣ Retrieve from ChromaDB First
#         context = retrieve_relevant_text(request.question)
#         if context and "No relevant data found" not in context:
#             return {"answer": context}

#         # 🔹 2️⃣ If ChromaDB fails, generate an LLM-based response
#         response = ollama.chat(model="mistral", messages=[
#             {"role": "system", "content": "You are an AI tutor that provides educational responses."},
#             {"role": "user", "content": request.question}
#         ])
#         print(f"🔍 Ollama Response: {response}")

#         llm_answer = response.get('message', {}).get('content', "")

#         # If LLM provides a reasonable answer, return it
#         if llm_answer and "I couldn't find a relevant answer." not in llm_answer:
#             return {"answer": llm_answer}

#         # 🔹 3️⃣ If LLM fails, try Web Search
#         try:
#             search_results = search_web(request.question + " latest research site:nature.com OR site:sciencedirect.com OR site:nasa.gov")
#             if search_results and search_results[0] != "No relevant results found.":
#                 return {"answer": f"🔎 Here's the latest research update:\n{search_results[0]}"}
#         except Exception as e:
#             logging.error(f"Web search error: {e}")

#         # 🔹 4️⃣ No Wikipedia fallback, just a final response
#         return {"answer": "I couldn't find a relevant answer from any sources. Try rephrasing your question."}

#     except Exception as e:
#         logging.error(f"Error processing request: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again later.")


# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     """Handles file uploads for study materials (PDF, DOCX, TXT)."""
#     try:
#         text = process_uploaded_file(file)
#         print(f"Extracted Text: {text}")  # Debugging step
#         store_text_in_chroma(text, file.filename)
#         return {"message": "File processed and stored successfully"} #"extracted_text": text
#     except ValueError as ve:
#         raise HTTPException(status_code=400, detail=str(ve))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# #uvicorn Backend.app:app --host 0.0.0.0 --port 8000 --reload






#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




# @app.get("/query")
# async def get_answer(query: str):
#     """Handles user queries and retrieves relevant information."""
#     logging.info(f"Received query: {query}")

#     query = query.lower().strip()
#     if query in COMMON_RESPONSES:
#         return {"response": COMMON_RESPONSES[query]}

#     if len(query) < 3:  
#         return {"response": "Could you provide more details?"}

#     # chroma_result = await retrieve_relevant_text(query, model)
#     # wiki_result = await async_wikipedia_search(query)
#     # google_result = await search_google(query)
#     # web_scrape_result = await scrape_and_summarize(query)

#     # response_parts = []
#     # if chroma_result:
#     #     response_parts.append(f"📚 **ChromaDB Result:**\n{chroma_result}")
#     # if wiki_result:
#     #     response_parts.append(f"🌍 **Wikipedia:**\n{wiki_result}")
#     # if google_result:
#     #     response_parts.append(f"🔎 **Google Results:**\n" + "\n".join(google_result))
#     # if web_scrape_result:
#     #     response_parts.append(f"📄 **Extracted Summary:**\n{web_scrape_result}")

#     # if response_parts:
#     #     return {"response": "\n\n".join(response_parts)}

#     # return {"response": "❌ No relevant data found. Try asking a different educational question. 😊"}

#     response_parts = []
    
#     try:
#         chroma_result = await retrieve_relevant_text(query, model)
#         if chroma_result:
#             response_parts.append(f"📚 **ChromaDB Result:**\n{chroma_result}")
#     except Exception as e:
#         logging.error(f"❌ Error retrieving ChromaDB result: {e}", exc_info=True)



#     try:
#        wiki_result = await async_wikipedia_search(query)

#        if wiki_result:
#          print(f"📝 Storing Wikipedia Data: {wiki_result}")
 
#         # Ensure context_parts is initialized properly
#          if 'context_parts' not in locals():
#             context_parts = []

#          store_text_in_chroma(wiki_result, "Wikipedia", model)
#          context_parts.append(wiki_result)
#          response_parts.append(f"🌍 **Wikipedia:**\n{wiki_result}")
#        else:
#         print(f"⚠️ No relevant Wikipedia result found for '{query}'")
#     except Exception as e:
#        print(f"❌ Error retrieving Wikipedia data: {e}")


#     # try:
#     #    wiki_result = await async_wikipedia_search(query)
    
#     #    if wiki_result:
#     #       print(f"📝 Storing Wikipedia Data: {wiki_result}")  # Debug print
#     #       store_text_in_chroma(wiki_result, "Wikipedia", model)  

#     #       if not isinstance(context_parts, list):  # Ensure list type
#     #          context_parts = []  
        
#     #       context_parts.append(wiki_result)  # Append Wikipedia result
#     #       response_parts.append(f"🌍 **Wikipedia:**\n{wiki_result}")
#     #    else:
#     #       print(f"⚠️ No relevant Wikipedia result found for '{query}'")
#     # except Exception as e:
#     #    print(f"❌ Error retrieving Wikipedia data: {e}")


    
#     try:
#         google_result = await search_google(query,model)
#         if google_result:
#             logging.info(f"🔎 Google Results: {google_result}")
#             response_parts.append(f"🔎 **Google Results:**\n" + "\n".join(google_result))
#     except Exception as e:
#         logging.error(f"Error in retrieving Google results: {e}")
#         google_result = None
    
#     try:
#         web_scrape_result = await scrape_and_summarize(query)
#         if web_scrape_result:
#             response_parts.append(f"📄 **Extracted Summary:**\n{web_scrape_result}")
#     except Exception as e:
#         logging.error(f"Error in scraping and summarizing: {e}")

#     if response_parts:
#         return {"response": "\n\n".join(response_parts)}

#     if not response_parts:
#        return {"response": "I couldn't find relevant information. Try rephrasing your query!"}


#     # 🔹 Step 6: Generate Final Answer Using LLM
#     try:
#         combined_context = "\n\n".join(context_parts)
#         final_prompt = f"Use the following context to answer the question:\n\n{combined_context}\n\nQuestion: {query}\nAnswer:"
        
#         final_answer = generate_llm_response.generate(final_prompt)

#         response_parts.append(f"💡 **Final Answer:**\n{final_answer}")

#     except Exception as e:
#         logging.error(f"❌ Error generating final LLM response: {e}", exc_info=True)
#         final_answer = "I couldn't generate a response."

#     # 🔹 Step 7: Return the Response
#     return {"response": "\n\n".join(response_parts)}