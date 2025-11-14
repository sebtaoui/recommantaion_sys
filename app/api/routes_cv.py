from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from app.core.config import collection, collection_embedded_data, collection_faiss, collection_data_cleaning, settings
from app.database.schemas import all_tasks_search
from app.services.extraction import extract_text_from_cv
from app.services.structuring import extract_structured_info, extract_experience_summary
from app.matching_search.embedding import build_text_from_json
from app.core.llm_api import extract_phone_number_from_text
from app.database.models import CVPhone
from bson import ObjectId
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache
import zipfile
import io
import faiss
import numpy as np
import pickle
import json
import re
import os
import math

router = APIRouter()

embedding_backend = settings.embedding_model
cross_encoder = settings.cross_encoder

FAISS_INDEX_PATH = "cv_index.faiss"
ID_MAP_PATH = "id_map.pkl"

KEYWORD_BOOST_FACTOR = 0.05
KEYWORD_BOOST_CAP = 0.4
MAX_KEYWORD_WEIGHT = 5.0
EXPERIENCE_BOOST_FACTOR = 0.15
YEARS_NORMALIZATION_CAP = 12.0
SENIORITY_WEIGHTS = {"junior": 0.35, "mid": 0.65, "senior": 1.0}

FAISS_PRESELECTION_K = max(1, getattr(settings, "faiss_preselection_k", 100))
DEFAULT_FINAL_TOP_K = 10
EMBEDDING_SCORE_WEIGHT = max(0.0, getattr(settings, "fusion_embedding_weight", 0.45))
RERANKER_SCORE_WEIGHT = max(0.0, getattr(settings, "fusion_cross_encoder_weight", 0.45))
KEYWORD_SCORE_WEIGHT = max(0.0, getattr(settings, "fusion_keyword_weight", 0.05))
EXPERIENCE_SCORE_WEIGHT = max(0.0, getattr(settings, "fusion_experience_weight", 0.05))


def normalize_phone_number(phone: Optional[str]) -> Optional[str]:
    """
    Normalize phone number by removing spaces, dashes, parentheses, and other formatting.
    Returns None if phone is empty or invalid.
    """
    if not phone:
        return None
    
    # Convert to string and strip whitespace
    phone_str = str(phone).strip()
    
    # Remove common formatting characters (spaces, dashes, parentheses, dots, plus signs)
    normalized = re.sub(r'[\s\-\(\)\.]', '', phone_str)
    
    # Remove leading + if present (country code indicator)
    normalized = normalized.lstrip('+')
    
    # Keep only digits
    normalized = re.sub(r'\D', '', normalized)
    
    # Return None if too short (less than 6 digits) or empty
    if len(normalized) < 6:
        return None
    
    # For very long numbers, keep only the last 10-12 digits (local number)
    # This handles cases where country codes are included
    if len(normalized) > 12:
        normalized = normalized[-12:]
    
    return normalized


def check_duplicate_phone(phone: Optional[str]) -> bool:
    """
    Check if a CV with the same phone number already exists in the database.
    Returns True if duplicate found, False otherwise.
    """
    if not phone:
        return False
    
    normalized_phone = normalize_phone_number(phone)
    if not normalized_phone:
        return False
    
    # Get all CVs with phone numbers
    existing_cvs = collection.find({
        "phone": {"$exists": True, "$ne": None, "$ne": ""}
    })
    
    # Check each CV's phone number after normalization
    for cv in existing_cvs:
        existing_phone = cv.get("phone")
        if existing_phone:
            existing_normalized = normalize_phone_number(existing_phone)
            if existing_normalized and existing_normalized == normalized_phone:
                return True
    
    return False


def process_cv_file(file_bytes: bytes, filename: str) -> Tuple[Dict[str, Any], int, int]:
    """Process a single CV file and return (result_dict, processed_increment, skipped_increment)."""
    try:
        print(f"Processing: {filename}")

        extracted_text = extract_text_from_cv(file_bytes, filename)

        if not extracted_text.strip():
            print(f"Skipping empty CV: {filename}")
            return (
                {
                    "filename": filename,
                    "status": "skipped",
                    "reason": "Empty CV",
                },
                0,
                1,
            )

        cv_info_result = extract_structured_info(extracted_text)
        if isinstance(cv_info_result, dict) and "error" in cv_info_result:
            print(f"Extraction error for {filename}: {cv_info_result['error']}")
            return (
                {
                    "filename": filename,
                    "status": "error",
                    "error": cv_info_result["error"],
                },
                0,
                1,
            )

        cv_data = cv_info_result["model"]
        info_warnings = cv_info_result.get("warnings", [])

        # Check for duplicate phone number before insertion
        phone_number = cv_data.phone
        if phone_number and check_duplicate_phone(phone_number):
            print(
                f"Skipping duplicate CV (same phone number): {filename} - Phone: {phone_number}"
            )
            return (
                {
                    "filename": filename,
                    "status": "skipped",
                    "reason": f"Duplicate phone number: {phone_number}",
                },
                0,
                1,
            )

        structured_dict = cv_data.model_dump()
        if info_warnings:
            structured_dict["validationWarnings"] = info_warnings
        structured_dict["filename"] = filename
        insert_result = collection.insert_one(structured_dict)
        inserted_id = insert_result.inserted_id

        summary_result = extract_experience_summary(extracted_text, str(inserted_id))
        if isinstance(summary_result, dict) and "error" in summary_result:
            print(f"Summary extraction error for {filename}: {summary_result['error']}")
            return (
                {
                    "filename": filename,
                    "status": "error",
                    "error": summary_result["error"],
                },
                0,
                1,
            )

        cv_summary = summary_result["model"]
        summary_warnings = summary_result.get("warnings", [])

        summary_dict = cv_summary.model_dump()
        if summary_warnings:
            summary_dict["validationWarnings"] = summary_warnings
        summary_dict["filename"] = filename
        summary_dict["original_cv_id"] = str(inserted_id)

        embedding_text = build_text_from_json(summary_dict, experience_weight=3.0)
        embedding_vector = embedding_backend.encode(embedding_text, normalize=True)
        embedding_array = np.array(embedding_vector, dtype="float32")

        summary_dict["embedding"] = embedding_array.tolist()
        insert_embedding_result = collection_embedded_data.insert_one(summary_dict)
        embedding_id = insert_embedding_result.inserted_id

        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(ID_MAP_PATH):
            index = faiss.read_index(FAISS_INDEX_PATH)
            with open(ID_MAP_PATH, "rb") as f:
                id_map = pickle.load(f)
        else:
            dim = len(embedding_array)
            index = faiss.IndexFlatIP(dim)
            id_map = []

        vector = np.array(embedding_array, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vector)
        index.add(vector)
        id_map.append(str(embedding_id))

        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(ID_MAP_PATH, "wb") as f:
            pickle.dump(id_map, f)

        return (
            {
                "filename": filename,
                "status": "success",
                "cv_id": str(inserted_id),
                "embedding_id": str(embedding_id),
            },
            1,
            0,
        )

    except Exception as cv_error:
        print(f"Error processing {filename}: {cv_error}")
        return (
            {
                "filename": filename,
                "status": "error",
                "error": str(cv_error),
            },
            0,
            1,
        )

_score_weight_sum = EMBEDDING_SCORE_WEIGHT + RERANKER_SCORE_WEIGHT + KEYWORD_SCORE_WEIGHT + EXPERIENCE_SCORE_WEIGHT
if _score_weight_sum <= 0:
    EMBEDDING_SCORE_WEIGHT = 0.45
    RERANKER_SCORE_WEIGHT = 0.45
    KEYWORD_SCORE_WEIGHT = 0.05
    EXPERIENCE_SCORE_WEIGHT = 0.05
else:
    EMBEDDING_SCORE_WEIGHT /= _score_weight_sum
    RERANKER_SCORE_WEIGHT /= _score_weight_sum
    KEYWORD_SCORE_WEIGHT /= _score_weight_sum
    EXPERIENCE_SCORE_WEIGHT /= _score_weight_sum


def preprocess_query(raw_query: str) -> Dict[str, Any]:
    normalized = re.sub(r"\s+", " ", raw_query).strip()
    lowered = normalized.lower()
    tokens = [token for token in re.split(r"\W+", lowered) if len(token) >= 3]
    keywords = list(dict.fromkeys(tokens))

    years_requested = None
    years_match = re.search(r"(\d{1,2})\s*(?:\+?\s*)?(?:years?|ans)", lowered)
    if years_match:
        try:
            years_requested = int(years_match.group(1))
        except ValueError:
            years_requested = None

    seniority_hint = None
    if "senior" in lowered or (years_requested and years_requested >= 5):
        seniority_hint = "senior"
    elif "lead" in lowered or "expert" in lowered:
        seniority_hint = "senior"
    elif "intermediate" in lowered or "mid" in lowered:
        seniority_hint = "mid"
    elif "junior" in lowered:
        seniority_hint = "junior"

    focus_domains = []
    for domain in ("data", "ai", "ml", "cloud", "security", "devops", "frontend", "backend", "fullstack"):
        if domain in lowered and domain not in focus_domains:
            focus_domains.append(domain)

    rerank_query_parts = [normalized]
    if keywords:
        rerank_query_parts.append("Focus keywords: " + ", ".join(keywords[:20]))
    if seniority_hint:
        rerank_query_parts.append(f"Target seniority: {seniority_hint}")
    if years_requested:
        rerank_query_parts.append(f"Years required: {years_requested}")
    if focus_domains:
        rerank_query_parts.append("Domains: " + ", ".join(focus_domains))

    rerank_query = " | ".join(rerank_query_parts)

    return {
        "normalized_query": normalized,
        "lowered_query": lowered,
        "keywords": keywords,
        "years_requested": years_requested,
        "seniority_hint": seniority_hint,
        "focus_domains": focus_domains,
        "rerank_query": rerank_query
    }


def merge_keyword_weights(
    default_keywords: List[str],
    custom_weights: Optional[Dict[str, float]]
) -> Dict[str, float]:
    weights: Dict[str, float] = {kw: 1.0 for kw in default_keywords}
    if custom_weights:
        for keyword, weight in custom_weights.items():
            weights[keyword] = min(MAX_KEYWORD_WEIGHT, max(weight, 0.0))
    return weights


def build_reranker_text(candidate: Dict[str, Any]) -> str:
    parts: List[str] = []

    summary = candidate.get("summaryText")
    if summary:
        parts.append(f"Summary: {summary}")

    experience_details = candidate.get("experienceDetails")
    if experience_details:
        parts.append(f"Experience: {experience_details}")

    hard_skills = candidate.get("hardSkills") or []
    if hard_skills:
        parts.append("Hard skills: " + ", ".join(hard_skills))

    soft_skills = candidate.get("softSkills") or []
    if soft_skills:
        parts.append("Soft skills: " + ", ".join(soft_skills))

    education = candidate.get("education")
    if education:
        parts.append(f"Education: {education}")

    total_years = candidate.get("totalYearsExperience")
    if total_years:
        parts.append(f"Total experience: {total_years} years")

    return " | ".join(parts)


def sigmoid(x: float) -> float:
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


@lru_cache(maxsize=2048)
def get_cross_encoder_score(query_text: str, candidate_text: str) -> float:
    """
    Lightweight LRU cache to avoid recomputing cross-encoder scores
    for identical (query, candidate) pairs during interactive usage.
    """
    scores = cross_encoder.predict([(query_text, candidate_text)])
    if isinstance(scores, np.ndarray):
        scores = scores.tolist()
    return float(scores[0])


class RecommendationRequest(BaseModel):
    query: str = Field(..., description="Search query describing the desired candidate profile.")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of candidates to return.")
    min_similarity: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum cosine similarity threshold (0-1).")
    keyword_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional mapping of keyword -> importance weight (>=0)."
    )
    experience_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Relative importance of the experience component (>=0)."
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Query cannot be empty.")
        return value.strip()

    @field_validator("keyword_weights")
    @classmethod
    def sanitize_keyword_weights(cls, value: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if value is None:
            return value
        sanitized: Dict[str, float] = {}
        for key, weight in value.items():
            normalized_key = key.strip().lower()
            if not normalized_key:
                continue
            try:
                weight_value = float(weight)
            except (TypeError, ValueError):
                continue
            if weight_value <= 0:
                continue
            sanitized[normalized_key] = min(weight_value, MAX_KEYWORD_WEIGHT)
        return sanitized or None

@router.post("/upload-cv-batch")
async def upload_cv_batch(zip_file: UploadFile = File(...)):
    """
    Upload a ZIP folder containing multiple CVs (up to 400+).
    Extracts text, analyzes experience, generates embeddings, and stores in MongoDB + FAISS.
    """
    try:
        zip_bytes = await zip_file.read()
        results = []
        processed_count = 0
        skipped_count = 0

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for file_info in zf.infolist():
                try:
                    if file_info.is_dir():
                        continue

                    file_bytes = zf.read(file_info.filename)
                    filename = file_info.filename

                    result_entry, processed_inc, skipped_inc = process_cv_file(file_bytes, filename)

                    if result_entry:
                        results.append(result_entry)
                    processed_count += processed_inc
                    skipped_count += skipped_inc

                except Exception as cv_error:
                    print(f"Error processing {file_info.filename}: {cv_error}")
                    skipped_count += 1
                    results.append({
                        "filename": file_info.filename,
                        "status": "error",
                        "error": str(cv_error)
                    })
                    continue

        return JSONResponse(content={
            "status": "completed",
            "total_processed": processed_count,
            "total_skipped": skipped_count,
            "message": f"Successfully processed {processed_count} CVs, skipped {skipped_count}",
            "results": results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch upload error: {str(e)}")


@router.post("/upload-cv")
async def upload_single_cv(file: UploadFile = File(...)):
    """
    Upload a single CV file (PDF, DOCX, image) and process it through the pipeline.
    """
    try:
        file_bytes = await file.read()
        result_entry, processed_inc, skipped_inc = process_cv_file(file_bytes, file.filename)

        return JSONResponse(content={
            "status": "completed" if processed_inc else "skipped",
            "total_processed": processed_inc,
            "total_skipped": skipped_inc,
            "message": (
                "CV processed successfully"
                if processed_inc
                else "CV skipped or failed"
            ),
            "results": [result_entry] if result_entry else [],
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Single upload error: {str(e)}")

@router.post("/recommend-candidates")
async def recommend_candidates(request: RecommendationRequest):
    """
    Recommend top K candidates based on a query prompt.
    Pipeline: embedding search → FAISS preselection → cross-encoder re-ranking → score fusion.
    """
    try:
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(ID_MAP_PATH):
            raise HTTPException(status_code=404, detail="No CVs indexed yet. Please upload CVs first.")

        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(ID_MAP_PATH, "rb") as f:
            id_map = pickle.load(f)

        if index.ntotal == 0:
            raise HTTPException(status_code=404, detail="FAISS index is empty")

        query_context = preprocess_query(request.query)
        processed_query = query_context["normalized_query"]
        rerank_query = query_context["rerank_query"]

        keyword_weights = merge_keyword_weights(
            query_context["keywords"],
            request.keyword_weights
        )

        final_top_k = max(1, min(request.top_k, DEFAULT_FINAL_TOP_K))
        experience_weight = max(0.0, request.experience_weight)

        # Encode query to embedding
        query_embedding = embedding_backend.encode(processed_query, normalize=True)
        query_vector = np.array(query_embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query_vector)

        # Stage 1: FAISS preselection (top N candidates)
        search_k = min(max(FAISS_PRESELECTION_K, final_top_k), index.ntotal)
        scores, indices = index.search(query_vector, search_k)

        preselected_candidates = []
        filtered_count = 0
        
        def get_score_quality(score: float) -> str:
            """Interpret similarity score quality"""
            if score >= 0.7:
                return "very_relevant"
            elif score >= 0.5:
                return "relevant"
            elif score >= 0.3:
                return "moderately_relevant"
            else:
                return "low_relevance"
        
        for score, idx in zip(scores[0], indices[0]):
            if score < request.min_similarity:
                filtered_count += 1
                continue
            
            if idx < len(id_map):
                cv_embedding_id = id_map[idx]
                cv_doc = collection_embedded_data.find_one({"_id": ObjectId(cv_embedding_id)})
                
                if cv_doc:
                    cv_doc.pop("embedding", None)
                    cv_doc["_id"] = str(cv_doc["_id"])
                    
                    original_cv_id = cv_doc.get("original_cv_id")
                    if original_cv_id:
                        full_cv = collection.find_one({"_id": ObjectId(original_cv_id)})
                        if full_cv:
                            full_cv["_id"] = str(full_cv["_id"])
                            cv_doc["full_cv_details"] = full_cv
                    preselected_candidates.append({
                        "candidate": cv_doc,
                        "similarity_score": float(score),
                        "embedding_score": float(score),
                        "score_quality": get_score_quality(float(score)),
                        "keyword_matches": [],
                        "keyword_boost": 0.0,
                        "experience_signal": 0.0,
                        "experience_boost": 0.0,
                        "reranker_score": 0.0,
                        "reranker_probability": 0.0,
                        "combined_score": 0.0
                    })

        if not preselected_candidates:
            return JSONResponse(content={
                "query": processed_query,
                "top_k_requested": final_top_k,
                "min_similarity_threshold": request.min_similarity,
                "total_candidates_found": 0,
                "total_filtered_out": filtered_count,
                "ranking_method": {
                    "base": "faiss_cosine_similarity",
                    "note": "No candidates passed the similarity threshold."
                },
                "recommendations": []
            })

        # Stage 2: Cross-encoder re-ranking (with LRU cache on identical pairs)
        candidate_rerank_texts = [
            build_reranker_text(entry["candidate"]) for entry in preselected_candidates
        ]
        reranker_scores = [
            get_cross_encoder_score(rerank_query, candidate_text)
            for candidate_text in candidate_rerank_texts
        ]

        # Stage 3: Score fusion
        recommendations = []
        for entry, rerank_score in zip(preselected_candidates, reranker_scores):
            candidate = entry["candidate"]
            rec = entry
            matched_keywords: Dict[str, float] = {}

            if candidate and keyword_weights:
                experience_text = candidate.get("experienceDetails", "") or ""
                summary_text = candidate.get("summaryText", "") or ""
                hard_skills = " ".join(candidate.get("hardSkills", []) or [])
                soft_skills = " ".join(candidate.get("softSkills", []) or [])

                searchable_text = f"{experience_text} {summary_text} {hard_skills} {soft_skills}".lower()

                for keyword, weight in keyword_weights.items():
                    if keyword and keyword in searchable_text:
                        matched_keywords[keyword] = weight

            keyword_boost = sum(
                KEYWORD_BOOST_FACTOR * min(MAX_KEYWORD_WEIGHT, weight)
                for weight in matched_keywords.values()
            )
            keyword_boost = min(keyword_boost, KEYWORD_BOOST_CAP)

            rec["keyword_matches"] = [
                {"keyword": keyword, "weight": round(weight, 4)}
                for keyword, weight in sorted(matched_keywords.items())
            ]
            rec["keyword_boost"] = round(keyword_boost, 4)

            experience_years = candidate.get("totalYearsExperience")
            experience_level = (candidate.get("experienceLevel") or "").lower()

            try:
                years_value = float(experience_years) if experience_years is not None else 0.0
            except (TypeError, ValueError):
                years_value = 0.0

            normalized_years = min(max(years_value, 0.0) / YEARS_NORMALIZATION_CAP, 1.0)
            seniority_signal = SENIORITY_WEIGHTS.get(experience_level, 0.0)
            experience_signal = max(normalized_years, seniority_signal)
            experience_boost = EXPERIENCE_BOOST_FACTOR * experience_weight * experience_signal

            rec["experience_signal"] = round(experience_signal, 4)
            rec["experience_boost"] = round(experience_boost, 4)

            embedding_component = max(0.0, min(1.0, (rec["similarity_score"] + 1.0) / 2.0))
            reranker_probability = sigmoid(float(rerank_score))

            keyword_component = keyword_boost / KEYWORD_BOOST_CAP if KEYWORD_BOOST_CAP > 0 else 0.0
            experience_component = min(1.0, experience_signal * max(experience_weight, 0.0))

            combined_score = (
                EMBEDDING_SCORE_WEIGHT * embedding_component +
                RERANKER_SCORE_WEIGHT * reranker_probability +
                KEYWORD_SCORE_WEIGHT * keyword_component +
                EXPERIENCE_SCORE_WEIGHT * experience_component
            )

            rec.update({
                "embedding_component": round(embedding_component, 4),
                "reranker_score": round(float(rerank_score), 4),
                "reranker_probability": round(reranker_probability, 4),
                "keyword_component": round(keyword_component, 4),
                "experience_component": round(experience_component, 4),
                "combined_score": round(combined_score, 4)
            })

            recommendations.append(rec)

        # Re-rank by boosted similarity
        recommendations.sort(key=lambda r: r["combined_score"], reverse=True)
        for idx, rec in enumerate(recommendations, start=1):
            rec["rank"] = idx

        # Trim to requested top_k in case boosting re-ordered list beyond limit
        recommendations = recommendations[:final_top_k]

        # Calculate statistics
        avg_score = float(np.mean([r["similarity_score"] for r in recommendations])) if recommendations else 0.0
        max_score = float(np.max([r["similarity_score"] for r in recommendations])) if recommendations else 0.0
        min_score = float(np.min([r["similarity_score"] for r in recommendations])) if recommendations else 0.0
        combined_avg = float(np.mean([r["combined_score"] for r in recommendations])) if recommendations else 0.0
        combined_max = float(np.max([r["combined_score"] for r in recommendations])) if recommendations else 0.0
        combined_min = float(np.min([r["combined_score"] for r in recommendations])) if recommendations else 0.0
        reranker_avg = float(np.mean([r["reranker_probability"] for r in recommendations])) if recommendations else 0.0

        return JSONResponse(content={
            "query": processed_query,
            "top_k_requested": final_top_k,
            "min_similarity_threshold": request.min_similarity,
            "experience_weight": experience_weight,
            "query_analysis": {
                "normalized_query": query_context["normalized_query"],
                "keywords": query_context["keywords"],
                "seniority_hint": query_context["seniority_hint"],
                "years_requested": query_context["years_requested"],
                "focus_domains": query_context["focus_domains"]
            },
            "preselection_size": len(preselected_candidates),
            "total_candidates_found": len(recommendations),
            "total_filtered_out": filtered_count,
            "score_statistics": {
                "average": round(avg_score, 4),
                "maximum": round(max_score, 4),
                "minimum": round(min_score, 4),
                "combined_average": round(combined_avg, 4),
                "combined_maximum": round(combined_max, 4),
                "combined_minimum": round(combined_min, 4),
                "reranker_probability_average": round(reranker_avg, 4)
            },
            "ranking_method": {
                "base": "faiss_cosine_similarity",
                "post_processing": "keyword_and_experience_weighting",
                "faiss_preselection_k": min(search_k, index.ntotal),
                "embedding_model": getattr(settings, "model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                "cross_encoder_model": settings.reranker_model_name if hasattr(settings, "reranker_model_name") else "unknown",
                "keyword_boost_factor": KEYWORD_BOOST_FACTOR,
                "keyword_boost_cap": KEYWORD_BOOST_CAP,
                "experience_boost_factor": EXPERIENCE_BOOST_FACTOR,
                "fusion_weights": {
                    "embedding": EMBEDDING_SCORE_WEIGHT,
                    "cross_encoder": RERANKER_SCORE_WEIGHT,
                    "keywords": KEYWORD_SCORE_WEIGHT,
                    "experience": EXPERIENCE_SCORE_WEIGHT
                },
                "keywords_used": [
                    {"keyword": keyword, "weight": round(weight, 4)}
                    for keyword, weight in sorted(keyword_weights.items())
                ]
            },
            "recommendations": recommendations,
            "note": "Combined score fusion of embedding similarity, cross-encoder relevance, keywords, and experience signals."
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

@router.get("/")
async def get_all_embedded_cvs():
    """View all processed CVs with embeddings"""
    data = collection_embedded_data.find()
    return all_tasks_search(data)

@router.get("/stats")
async def get_stats():
    """Get statistics about the CV database"""
    total_cvs = collection.count_documents({})
    total_embedded = collection_embedded_data.count_documents({})
    
    index_size = 0
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        index_size = index.ntotal
    
    return {
        "total_cvs_stored": total_cvs,
        "total_cvs_embedded": total_embedded,
        "faiss_index_size": index_size,
        "ready_for_search": index_size > 0
    }

@router.delete("/clear-all")
def clear_all_data():
    """Clear all CV data and FAISS index"""
    try:
        cv_result = collection.delete_many({})
        embedded_result = collection_embedded_data.delete_many({})
        faiss_result = collection_faiss.delete_many({})
        
        if os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH)
        if os.path.exists(ID_MAP_PATH):
            os.remove(ID_MAP_PATH)
        get_cross_encoder_score.cache_clear()
        
        return {
            "message": "All data cleared successfully",
            "cvs_deleted": cv_result.deleted_count,
            "embedded_deleted": embedded_result.deleted_count,
            "faiss_deleted": faiss_result.deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing data: {e}")

@router.post("/data-preprocessing", response_model=List[CVPhone])
async def extract_phone_numbers(zip_file: UploadFile = File(...)):
    """
    Extract phone numbers from CVs for data cleaning/preprocessing.
    """
    zip_bytes = await zip_file.read()
    results: List[CVPhone] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for file_info in zf.infolist():
            if file_info.is_dir():
                continue

            file_bytes = zf.read(file_info.filename)
            text = extract_text_from_cv(file_bytes, file_info.filename)

            if not text.strip():
                continue

            phone_data = extract_phone_number_from_text(text)
            phone_number = phone_data.get("phone_number")

            record = {"filename": file_info.filename, "phone_number": phone_number}
            collection_data_cleaning.insert_one(record)

            results.append(CVPhone(filename=file_info.filename, phone_number=phone_number))

    return results
