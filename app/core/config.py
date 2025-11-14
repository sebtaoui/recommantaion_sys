import json
import os 
from pymongo import MongoClient
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from app.core.embedding_models import build_embedding_backend


def get_float_env(key: str, default: float) -> float:
    raw_value = os.getenv(key)
    if raw_value is None:
        return float(default)
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return float(default)


def get_int_env(key: str, default: int) -> int:
    raw_value = os.getenv(key)
    if raw_value is None:
        return int(default)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return int(default)

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

    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local")
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    reranker_model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    faiss_preselection_k = get_int_env("FAISS_PRESELECTION_K", 100)
    fusion_embedding_weight = get_float_env("FUSION_EMBEDDING_WEIGHT", 0.45)
    fusion_cross_encoder_weight = get_float_env("FUSION_CROSS_ENCODER_WEIGHT", 0.45)
    fusion_keyword_weight = get_float_env("FUSION_KEYWORD_WEIGHT", 0.05)
    fusion_experience_weight = get_float_env("FUSION_EXPERIENCE_WEIGHT", 0.05)
    calibration_profile_path = os.getenv("CALIBRATION_PROFILE_PATH", "calibration/weights.json")

    if os.path.exists(calibration_profile_path):
        try:
            with open(calibration_profile_path, "r", encoding="utf-8") as profile_file:
                profile_data = json.load(profile_file)
            faiss_preselection_k = profile_data.get("faiss_preselection_k", faiss_preselection_k)
            fusion_embedding_weight = profile_data.get(
                "fusion", {}
            ).get("embedding", fusion_embedding_weight)
            fusion_cross_encoder_weight = profile_data.get(
                "fusion", {}
            ).get("cross_encoder", fusion_cross_encoder_weight)
            fusion_keyword_weight = profile_data.get("fusion", {}).get("keywords", fusion_keyword_weight)
            fusion_experience_weight = profile_data.get(
                "fusion", {}
            ).get("experience", fusion_experience_weight)
        except Exception as calibration_error:
            print(f"[Calibration] Unable to load calibration profile: {calibration_error}")

    embedding_model = build_embedding_backend(
        embedding_provider,
        model_name=model_name,
        together_api_key=together_ai_API_KEY,
        together_base_url=together_ai_BASE_URL,
    )
    cross_encoder = CrossEncoder(reranker_model_name)
    warmed_up = False

    def warmup_models(self) -> None:
        """
        Run a lightweight warm-up pass on the embedding model and cross-encoder
        to avoid cold-start latency on the first user request.
        """
        try:
            self.embedding_model.encode("warm-up embedding", normalize=True)
            self.cross_encoder.predict([
                ("warm-up query", "warm-up candidate text focusing on experience and skills.")
            ])
            self.warmed_up = True
            print("[Warmup] Embedding model and cross-encoder warmed up successfully.")
        except Exception as exc:
            self.warmed_up = False
            print(f"[Warmup] Warning: unable to warm up models ({exc}).")

settings = Settings()
