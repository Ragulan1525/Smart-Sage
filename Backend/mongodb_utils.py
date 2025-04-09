# db/mongo_utils.py
from pymongo import MongoClient
from datetime import datetime
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["smart_sage_db"]
chat_collection = db["chat_history"]

def save_message(chat_id, role, content):
    """Save a single message with role and content."""
    chat_collection.insert_one({
        "chat_id": chat_id,
        "timestamp": datetime.utcnow(),
        "role": role,
        "content": content
    })

def get_chat_history(chat_id):
    """Fetch all messages in chronological order."""
    messages = chat_collection.find({"chat_id": chat_id}).sort("timestamp", 1)
    return [{"role": m["role"], "content": m["content"]} for m in messages]
