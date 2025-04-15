

# import streamlit as st
# import requests
# import json
# import time

# # FastAPI Backend URL
# API_URL = "http://127.0.0.1:8000"

# # Page Configurations
# st.set_page_config(page_title="Smart Sage - AI Study Assistant", layout="wide")

# # Custom CSS for Responsive UI and Animations
# st.markdown(
#     """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
#     html, body, [class*="css"] {
#         font-family: 'Inter', sans-serif;
#     }

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
#         transition: all 0.3s ease;
#     }
#     .user-message {
#         background-color: #e0e0e0;
#         color: black;
#         text-align: right;
#         padding: 10px;
#     }
#     .bot-message {
#         background-color: #4CAF50;
#         color: white;
#         padding: 10px;
#     }
#     .loading-dots span {
#         animation: blink 1.2s infinite;
#         animation-fill-mode: both;
#         font-size: 1.5rem;
#         padding: 0 2px;
#     }
#     .loading-dots span:nth-child(2) {
#         animation-delay: 0.2s;
#     }
#     .loading-dots span:nth-child(3) {
#         animation-delay: 0.4s;
#     }
#     @keyframes blink {
#         0%, 80%, 100% {
#             opacity: 0;
#         }
#         40% {
#             opacity: 1;
#         }
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # Initialize chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "bot_typing" not in st.session_state:
#     st.session_state.bot_typing = False

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

# # Get Answer Button
# if st.button("🔍 Get Answer"):
#     if user_query:
#         # Append user message
#         st.session_state.chat_history.append({"role": "user", "message": user_query})
#         st.session_state.bot_typing = True
#         st.rerun()

# # Simulated Bot Typing (after rerun)
# if st.session_state.bot_typing:
#     with st.container():
#         st.markdown(
#             '<div class="message-container bot-message"><span class="loading-dots">Thinking<span>.</span><span>.</span><span>.</span></span></div>',
#             unsafe_allow_html=True,
#         )

#     # Delay to simulate loading
#     with st.spinner("Smart Sage is generating a response..."):
#         time.sleep(1.5)  # simulate API delay
#         last_query = [m["message"] for m in st.session_state.chat_history if m["role"] == "user"][-1]
#         response = requests.get(f"{API_URL}/query", params={"query": last_query})
#         if response.status_code == 200:
#             bot_reply = response.json().get("response", "No answer found.")
#         else:
#             bot_reply = f"❌ Error retrieving answer: {response.text}"
#         st.session_state.chat_history.append({"role": "bot", "message": bot_reply})
#         st.session_state.bot_typing = False
#         st.rerun()

# # Clear Chat Button
# if st.button("🗑️ Clear Chat"):import streamlit as st






import streamlit as st
import requests
import json
import time

API_URL = "http://127.0.0.1:8000"

# Page Settings
st.set_page_config(page_title="Smart Sage - AI Study Assistant", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Rubik', sans-serif;
    }

    .main {
        background-color: #f4f4f4;
    }

    .stTextInput>div>div>input {
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #ccc;
        background-color: #1f2937;
        color: white;
    }

    .message-container {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }

    .user-message {
        background-color: #e0e0e0;
        color: black;
        text-align: right;
        padding: 10px;
    }

    .bot-message {
        background-color: #1f2937;
        color: white;
        padding: 10px;
    }

    .smart-sage-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .loading-dots span {
        animation: blink 1.2s infinite;
        animation-fill-mode: both;
        font-size: 1.5rem;
        padding: 0 2px;
    }

    .loading-dots span:nth-child(2) {
        animation-delay: 0.2s;
    }

    .loading-dots span:nth-child(3) {
        animation-delay: 0.4s;
    }

    @keyframes blink {
        0%, 80%, 100% {
            opacity: 0;
        }
        40% {
            opacity: 1;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bot_typing" not in st.session_state:
    st.session_state.bot_typing = False
if "last_input" not in st.session_state:
    st.session_state.last_input = ""

# Sidebar - File Upload
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

# Title Section with Gradient Text
st.markdown('<h1 class="smart-sage-title">Smart Sage</h1>', unsafe_allow_html=True)
st.write("**Chat with Smart Sage to get relevant educational answers.**")

# Display Chat History
for chat in st.session_state.chat_history:
    with st.container():
        st.markdown(
            f'<div class="message-container {"user-message" if chat["role"] == "user" else "bot-message"}">'
            f'{chat["message"]}</div>',
            unsafe_allow_html=True,
        )

# Text Input (Triggers on Enter)
user_query = st.text_input("Ask something:", placeholder="e.g., Explain Neural Networks", key="user_input")

if user_query and user_query != st.session_state.last_input:
    st.session_state.last_input = user_query
    st.session_state.chat_history.append({"role": "user", "message": user_query})
    st.session_state.bot_typing = True
    st.rerun()

# Bot Typing and Response Handling
if st.session_state.bot_typing:
    with st.container():
        st.markdown(
            '<div class="message-container bot-message"><span class="loading-dots">Thinking<span>.</span><span>.</span><span>.</span></span></div>',
            unsafe_allow_html=True,
        )

    with st.spinner("Smart Sage is generating a response..."):
        time.sleep(1.5)
        last_query = [m["message"] for m in st.session_state.chat_history if m["role"] == "user"][-1]
        response = requests.get(f"{API_URL}/query", params={"query": last_query})
        if response.status_code == 200:
            bot_reply = response.json().get("response", "No answer found.")
        else:
            bot_reply = f"❌ Error retrieving answer: {response.text}"

        st.session_state.chat_history.append({"role": "bot", "message": bot_reply})
        st.session_state.bot_typing = False
        st.rerun()

# Clear Chat Button
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.last_input = ""
    st.session_state.bot_typing = False
    st.rerun()
