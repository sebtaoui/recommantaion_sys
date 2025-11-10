import os 
from pymongo import MongoClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "cv_recommendation_db")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

collection = db["cvs"]
collection_embedded_data = db["embedded_data"]
collection_faiss = db["faiss_collection"]
collection_data_cleaning = db["data_cleaning"]

class Settings:
    # GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    # GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    # GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

    together_ai_API_KEY = os.getenv("TOGETHER_API_KEY")
    together_ai_BASE_URL = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
    together_ai_MODEL = os.getenv("TOGETHER_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

settings = Settings()
