import json
from typing import Dict, List

from app.database.models import CVInfo, CVSummary
from app.core.llm_api import extract_cv_info_via_llm, analyze_cv_for_experience
from app.services.validators import post_process_cv_info, post_process_cv_summary
from pydantic import ValidationError


def extract_structured_info(text: str):
    try:
        structured_dict = extract_cv_info_via_llm(text)

        if not structured_dict or (isinstance(structured_dict, dict) and len(structured_dict) == 0):
            return {"error": "Empty response from LLM"}

        cv_data = CVInfo.model_validate(structured_dict)
        normalized_cv, warnings = post_process_cv_info(cv_data)

        return {
            "model": normalized_cv,
            "warnings": warnings,
        }
    except ValidationError as e:
        return {"error": f"Data does not match CVInfo schema: {e}", "raw": structured_dict}
    except Exception as e:
        return {"error": str(e)}


def extract_experience_summary(text: str, cv_id: str):
    try:
        structured_dict = analyze_cv_for_experience(text, cv_id)

        if not structured_dict or (
            isinstance(structured_dict, dict) and len(structured_dict) == 1 and "cv_id" in structured_dict
        ):
            return {"error": "Empty or invalid response from LLM"}

        cv_embedded = CVSummary.model_validate(structured_dict)
        normalized_summary, warnings = post_process_cv_summary(cv_embedded)

        return {
            "model": normalized_summary,
            "warnings": warnings,
        }
    except ValidationError as e:
        return {"error": f"Data does not match CVSummary schema: {e}", "raw": structured_dict}
    except Exception as e:
        return {"error": str(e)}
