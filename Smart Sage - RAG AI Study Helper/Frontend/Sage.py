import streamlit as st
import ollama
from backend.retrieval import retrieve_relevant_text

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit UI
st.title("🤖 Smart Sage - AI Study Helper")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
user_input = st.chat_input("Ask your study-related question...")

if user_input:
    # Add user input to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Retrieve relevant educational content
    retrieved_text = retrieve_relevant_text(user_input)

    # Generate response using LLaMA 3 (Ollama)
    ollama_response = ollama.chat(model="llama3", messages=[{"role": "user", "content": retrieved_text}])

    bot_reply = ollama_response["message"]["content"]

    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
