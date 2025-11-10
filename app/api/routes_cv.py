from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.core.config import collection, collection_embedded_data, collection_faiss, collection_data_cleaning, settings
from app.database.schemas import all_tasks_search
from app.services.extraction import extract_text_from_cv
from app.services.structuring import extract_structured_info, extract_experience_summary
from app.matching_search.embedding import build_text_from_json
from app.core.llm_api import extract_phone_number_from_text
from app.database.models import CVPhone
from bson import ObjectId
from typing import List
import zipfile
import io
import faiss
import numpy as np
import pickle
import json
import re

router = APIRouter()

model = settings.model

FAISS_INDEX_PATH = "cv_index.faiss"
ID_MAP_PATH = "id_map.pkl"

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
                    print(f"Processing: {filename}")

                    extracted_text = extract_text_from_cv(file_bytes, filename)
                    
                    if not extracted_text.strip():
                        print(f"Skipping empty CV: {filename}")
                        skipped_count += 1
                        continue

                    cv_data = extract_structured_info(extracted_text)
                    if isinstance(cv_data, dict) and "error" in cv_data:
                        print(f"Extraction error for {filename}: {cv_data['error']}")
                        skipped_count += 1
                        continue
                    
                    structured_dict = cv_data.model_dump()
                    structured_dict["filename"] = filename
                    insert_result = collection.insert_one(structured_dict)
                    inserted_id = insert_result.inserted_id

                    cv_summary = extract_experience_summary(extracted_text, str(inserted_id))
                    if isinstance(cv_summary, dict) and "error" in cv_summary:
                        print(f"Summary extraction error for {filename}: {cv_summary['error']}")
                        skipped_count += 1
                        continue

                    summary_dict = cv_summary.model_dump()
                    summary_dict["filename"] = filename
                    summary_dict["original_cv_id"] = str(inserted_id)

                    embedding_text = build_text_from_json(summary_dict, experience_weight=3.0)
                    embedding_array = model.encode(embedding_text, normalize_embeddings=True)

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

                    vector = np.array(embedding_array, dtype='float32').reshape(1, -1)
                    faiss.normalize_L2(vector)
                    index.add(vector)
                    id_map.append(str(embedding_id))

                    faiss.write_index(index, FAISS_INDEX_PATH)
                    with open(ID_MAP_PATH, "wb") as f:
                        pickle.dump(id_map, f)

                    processed_count += 1
                    results.append({
                        "filename": filename,
                        "status": "success",
                        "cv_id": str(inserted_id),
                        "embedding_id": str(embedding_id)
                    })

                except Exception as cv_error:
                    print(f"Error processing {filename}: {cv_error}")
                    skipped_count += 1
                    results.append({
                        "filename": filename,
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

@router.post("/recommend-candidates")
async def recommend_candidates(
    query: str = Form(...),
    top_k: int = Form(10),
    min_similarity: float = Form(0.3)
):
    """
    Recommend top K candidates based on a query prompt.
    Focuses heavily on professional experience matching.
    
    Args:
        query: Search query (e.g., "Python developer with 5 years experience")
        top_k: Number of results to return (default: 10, max: 100)
        min_similarity: Minimum similarity score threshold (default: 0.3, range: 0.0-1.0)
    """
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if top_k < 1 or top_k > 100:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
        
        if min_similarity < 0.0 or min_similarity > 1.0:
            raise HTTPException(status_code=400, detail="min_similarity must be between 0.0 and 1.0")

        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(ID_MAP_PATH):
            raise HTTPException(status_code=404, detail="No CVs indexed yet. Please upload CVs first.")

        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(ID_MAP_PATH, "rb") as f:
            id_map = pickle.load(f)

        if index.ntotal == 0:
            raise HTTPException(status_code=404, detail="FAISS index is empty")

        # Preprocess query: strip and normalize
        processed_query = query.strip()
        keyword_tokens = {token.lower() for token in re.split(r"\W+", processed_query) if len(token) >= 3}
        
        # Encode query to embedding
        query_embedding = model.encode(processed_query, normalize_embeddings=True)
        query_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)
        faiss.normalize_L2(query_vector)

        # Search for more candidates than requested to filter by threshold
        # Search for up to 3x the requested amount to ensure we get enough after filtering
        search_k = min(top_k * 3, index.ntotal)
        scores, indices = index.search(query_vector, search_k)

        recommendations = []
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
            # Apply similarity threshold filter
            if score < min_similarity:
                filtered_count += 1
                continue
            
            # Stop if we have enough results
            if len(recommendations) >= top_k:
                break
            
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

                    score_float = float(score)
                    recommendations.append({
                        "rank": len(recommendations) + 1,
                        "similarity_score": score_float,
                        "score_quality": get_score_quality(score_float),
                        "candidate": cv_doc,
                        "keyword_matches": [],
                        "boost": 0.0,
                        "boosted_similarity_score": score_float
                    })

        # Apply keyword-based re-ranking boost focused on experience
        keyword_boost_factor = 0.05  # boost per keyword match
        keyword_boost_cap = 0.25     # cap total boost

        for rec in recommendations:
            candidate = rec["candidate"]
            matched_keywords = []

            if candidate and keyword_tokens:
                experience_text = candidate.get("experienceDetails", "") or ""
                summary_text = candidate.get("summaryText", "") or ""
                hard_skills = " ".join(candidate.get("hardSkills", []) or [])
                soft_skills = " ".join(candidate.get("softSkills", []) or [])

                searchable_text = f"{experience_text} {summary_text} {hard_skills} {soft_skills}".lower()

                for keyword in keyword_tokens:
                    if keyword and keyword in searchable_text:
                        matched_keywords.append(keyword)

            boost = min(len(set(matched_keywords)) * keyword_boost_factor, keyword_boost_cap)
            boosted_score = rec["similarity_score"] + boost

            rec["keyword_matches"] = sorted(set(matched_keywords))
            rec["boost"] = round(boost, 4)
            rec["boosted_similarity_score"] = round(boosted_score, 4)

        # Re-rank by boosted similarity
        recommendations.sort(key=lambda r: r["boosted_similarity_score"], reverse=True)
        for idx, rec in enumerate(recommendations, start=1):
            rec["rank"] = idx

        # Trim to requested top_k in case boosting re-ordered list beyond limit
        recommendations = recommendations[:top_k]

        # Calculate statistics
        avg_score = float(np.mean([r["similarity_score"] for r in recommendations])) if recommendations else 0.0
        max_score = float(np.max([r["similarity_score"] for r in recommendations])) if recommendations else 0.0
        min_score = float(np.min([r["similarity_score"] for r in recommendations])) if recommendations else 0.0
        boosted_avg = float(np.mean([r["boosted_similarity_score"] for r in recommendations])) if recommendations else 0.0
        boosted_max = float(np.max([r["boosted_similarity_score"] for r in recommendations])) if recommendations else 0.0
        boosted_min = float(np.min([r["boosted_similarity_score"] for r in recommendations])) if recommendations else 0.0

        return JSONResponse(content={
            "query": processed_query,
            "top_k_requested": top_k,
            "min_similarity_threshold": min_similarity,
            "total_candidates_found": len(recommendations),
            "total_filtered_out": filtered_count,
            "score_statistics": {
                "average": round(avg_score, 4),
                "maximum": round(max_score, 4),
                "minimum": round(min_score, 4),
                "boosted_average": round(boosted_avg, 4),
                "boosted_maximum": round(boosted_max, 4),
                "boosted_minimum": round(boosted_min, 4)
            },
            "ranking_method": {
                "base": "faiss_cosine_similarity",
                "post_processing": "keyword_boost_re_rank",
                "keyword_boost_factor": keyword_boost_factor,
                "keyword_boost_cap": keyword_boost_cap,
                "keywords_used": sorted(list(keyword_tokens))
            },
            "recommendations": recommendations,
            "note": "Similarity scores range from 0.0 to 1.0. Results are re-ranked by boosting candidates whose experience text matches query keywords."
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

import os
