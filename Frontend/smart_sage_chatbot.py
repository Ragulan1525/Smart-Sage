# import streamlit as st
# import requests
# import json

# # FastAPI Backend URL
# API_URL = "http://127.0.0.1:8000"

# # Page Configurations
# st.set_page_config(page_title="Smart Sage - AI Study Assistant", layout="wide")

# # Custom CSS for Responsive UI
# st.markdown(
#     """
#     <style>
#     .main {
#         background-color: #f4f4f4;
#     }
#     .stTextInput>div>div>input {
#         border-radius: 10px;
#         padding: 10px;
#         border: 1px solid #ccc;
#     }
#     .message-container {
#         padding: 10px;
#         border-radius: 10px;
#         margin-bottom: 10px;
#     }
#     .user-message {
#         background-color: #4CAF50;
#         color: white;
#         text-align: right;
#     }
#     .bot-message {
#         background-color: #e0e0e0;
#         color: black;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # Initialize chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Sidebar for File Upload
# st.sidebar.title("📂 Upload Study Material")
# uploaded_file = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

# if uploaded_file:
#     st.sidebar.write("Processing file...")
#     files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
#     response = requests.post(f"{API_URL}/upload", files=files)

#     if response.status_code == 200:
#         st.sidebar.success("✅ File uploaded and indexed successfully!")
#     else:
#         st.sidebar.error(f"❌ Error processing file: {response.text}")

# # Chatbot UI
# st.title("📚 Smart Sage - AI Study Assistant")
# st.write("**Chat with Smart Sage to get relevant educational answers.**")

# # Display chat history
# for chat in st.session_state.chat_history:
#     with st.container():
#         st.markdown(
#             f'<div class="message-container {"user-message" if chat["role"] == "user" else "bot-message"}">'
#             f'{chat["message"]}</div>',
#             unsafe_allow_html=True,
#         )

# # User Input
# user_query = st.text_input("Ask something:", placeholder="e.g., Explain Neural Networks", key="user_input")

# if st.button("🔍 Get Answer"):
#     if user_query:
#         # Append user message to chat history
#         st.session_state.chat_history.append({"role": "user", "message": user_query})

#         # Send query to FastAPI backend
#         response = requests.get(f"{API_URL}/query", params={"query": user_query})

#         if response.status_code == 200:
#             bot_reply = response.json().get("response", "No answer found.")
#             st.session_state.chat_history.append({"role": "bot", "message": bot_reply})
#         else:
#             bot_reply = f"❌ Error retrieving answer: {response.text}"
#             st.session_state.chat_history.append({"role": "bot", "message": bot_reply})

#         # Refresh page to display chat history
#         st.experimental_rerun()
#     else:
#         st.warning("⚠️ Please enter a question.")

# # Clear Chat Button
# if st.button("🗑️ Clear Chat"):
#     st.session_state.chat_history = []
#     st.experimental_rerun()

# # Footer
# st.sidebar.markdown("---")
# st.sidebar.markdown("Developed by **Smart Sage Team**")






import streamlit as st
import requests
import json

# FastAPI Backend URL
API_URL = "http://127.0.0.1:8000"

# Page Configurations
st.set_page_config(page_title="Smart Sage - AI Study Assistant", layout="wide")

# Custom CSS for Responsive UI
st.markdown(
    """
    <style>
    .main {
        background-color: #f4f4f4;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #ccc;
    }
    .message-container {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #4CAF50;
        color: white;
        text-align: right;
        padding: 10px;
    }
    .bot-message {
        background-color: #e0e0e0;
        color: black;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for File Upload
st.sidebar.title("📂 Upload Study Material")
uploaded_file = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

if uploaded_file:
    st.sidebar.write("Processing file...")
    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    response = requests.post(f"{API_URL}/upload", files=files)

    if response.status_code == 200:
        st.sidebar.success("✅ File uploaded and indexed successfully!")
    else:
        st.sidebar.error(f"❌ Error processing file: {response.text}")

# Chatbot UI
st.title("📚 Smart Sage - AI Study Assistant")
st.write("**Chat with Smart Sage to get relevant educational answers.**")

# Display chat history
for chat in st.session_state.chat_history:
    with st.container():
        st.markdown(
            f'<div class="message-container {"user-message" if chat["role"] == "user" else "bot-message"}">'
            f'{chat["message"]}</div>',
            unsafe_allow_html=True,
        )

# User Input
user_query = st.text_input("Ask something:", placeholder="e.g., Explain Neural Networks", key="user_input")

if st.button("🔍 Get Answer"):
    if user_query:
        # Append user message to chat history
        st.session_state.chat_history.append({"role": "user", "message": user_query})

        # Send query to FastAPI backend (prioritizing ChromaDB retrieval)
        response = requests.get(f"{API_URL}/query", params={"query": user_query})

        if response.status_code == 200:
            bot_reply = response.json().get("response", "No answer found.")
        else:
            bot_reply = f"❌ Error retrieving answer: {response.text}"

        # Store bot response in chat history
        st.session_state.chat_history.append({"role": "bot", "message": bot_reply})

        # Refresh page to display chat history
        st.rerun()
    else:
        st.warning("⚠️ Please enter a question.")

# Clear Chat Button
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by **Smart Sage Team**")


#streamlit run smart_sage_chatbot.py