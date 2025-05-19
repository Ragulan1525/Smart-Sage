# from motor.motor_asyncio import AsyncIOMotorClient
# from datetime import datetime, timedelta
# import os
# from typing import List, Dict
# import logging

# # MongoDB Connection (Async - Motor)
# MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
# client = AsyncIOMotorClient(MONGO_URI)
# db = client["smart_sage_db"]
# chat_collection = db["chat_history"]
# conversation_cache = db["conversation_cache"]

# # Constants
# MAX_CONVERSATION_LENGTH = 20
# CACHE_TTL = 3600  # 1 hour

# async def save_message(chat_id: str, role: str, content: str):
#     try:
#         await chat_collection.insert_one({
#             "chat_id": chat_id,
#             "timestamp": datetime.utcnow(),
#             "role": role,
#             "content": content
#         })
#         await update_conversation_cache(chat_id, role, content)
#         return True
#     except Exception as e:
#         logging.error(f"Error saving message: {e}")
#         return False

# async def get_chat_history(chat_id: str, limit: int = MAX_CONVERSATION_LENGTH) -> List[Dict]:
#     try:
#         cached = await conversation_cache.find_one({"chat_id": chat_id})
#         if cached and "last_updated" in cached and (datetime.utcnow() - cached["last_updated"]).total_seconds() < CACHE_TTL:
#             return cached.get("messages", [])[-limit:]

#         cursor = chat_collection.find({"chat_id": chat_id}).sort("timestamp", 1).limit(limit)
#         messages_list = []
#         async for message in cursor:
#             messages_list.append({"role": message["role"], "content": message["content"]})

#         await conversation_cache.update_one(
#             {"chat_id": chat_id},
#             {
#                 "$set": {
#                     "messages": messages_list,
#                     "last_updated": datetime.utcnow()
#                 }
#             },
#             upsert=True
#         )
#         return messages_list

#     except Exception as e:
#         logging.error(f"Error fetching chat history: {e}")
#         return []

# async def update_conversation_cache(chat_id: str, role: str, content: str):
#     try:
#         await conversation_cache.update_one(
#             {"chat_id": chat_id},
#             {
#                 "$push": {
#                     "messages": {
#                         "$each": [{"role": role, "content": content}],
#                         "$slice": -MAX_CONVERSATION_LENGTH
#                     }
#                 },
#                 "$set": {"last_updated": datetime.utcnow()}
#             },
#             upsert=True
#         )
#     except Exception as e:
#         logging.error(f"Error updating conversation cache: {e}")

# async def get_conversation_summary(chat_id: str) -> str:
#     try:
#         messages = await get_chat_history(chat_id)
#         if not messages:
#             return ""
#         conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
#         return conversation_text[:500] + "..." if len(conversation_text) > 500 else conversation_text
#     except Exception as e:
#         logging.error(f"Error generating conversation summary: {e}")
#         return ""

# async def cleanup_old_messages(days: int = 30):
#     try:
#         cutoff_date = datetime.utcnow() - timedelta(days=days)
#         result = await chat_collection.delete_many({"timestamp": {"$lt": cutoff_date}})
#         logging.info(f"Cleaned up {result.deleted_count} messages older than {days} days")
#     except Exception as e:
#         logging.error(f"Error cleaning up old messages: {e}")



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


# db/mongo_utils.py

def save_instruction(chat_id, instruction):
    """Save special user instructions separately if needed."""
    chat_collection.insert_one({
        "chat_id": chat_id,
        "timestamp": datetime.utcnow(),
        "role": "instruction",
        "content": instruction
    })

def get_full_memory(chat_id):
    """Fetch both chat history and instructions."""
    messages = chat_collection.find({"chat_id": chat_id}).sort("timestamp", 1)
    memory = []
    for m in messages:
        memory.append({
            "role": m["role"],  # user / assistant / instruction
            "content": m["content"]
        })
    return memory
