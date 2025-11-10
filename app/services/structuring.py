import json
from app.database.models import CVInfo, CVSummary
from app.core.llm_api import extract_cv_info_via_llm, analyze_cv_for_experience
from pydantic import ValidationError
import re

def extract_structured_info(text: str):
    try:
        # extract_cv_info_via_llm returns a dict (already parsed by safe_json_parse)
        structured_dict = extract_cv_info_via_llm(text)
        
        # Check if it's an error dict (empty or contains error info)
        if not structured_dict or (isinstance(structured_dict, dict) and len(structured_dict) == 0):
            return {"error": "Empty response from LLM"}
        
        # Validate and convert to CVInfo model
        cv_data = CVInfo.model_validate(structured_dict)
        return cv_data
    except ValidationError as e:
        return {"error": f"Data does not match CVInfo schema: {e}", "raw": structured_dict}
    except Exception as e:
        return {"error": str(e)}

def extract_experience_summary(text: str, cv_id: str):
    try:
        # analyze_cv_for_experience returns a dict (already parsed by safe_json_parse)
        structured_dict = analyze_cv_for_experience(text, cv_id)
        
        # Check if it's an error dict (only contains fallback cv_id)
        if not structured_dict or (isinstance(structured_dict, dict) and len(structured_dict) == 1 and "cv_id" in structured_dict):
            return {"error": "Empty or invalid response from LLM"}
        
        # Validate and convert to CVSummary model
        cv_embedded = CVSummary.model_validate(structured_dict)
        return cv_embedded
    except ValidationError as e:
        return {"error": f"Data does not match CVSummary schema: {e}", "raw": structured_dict}
    except Exception as e:
        return {"error": str(e)}
