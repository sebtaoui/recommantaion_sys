from app.core.config import settings
from openai import OpenAI
import json
import re
import httpx

# Configuration des timeouts pour éviter les erreurs de connexion
_timeout = httpx.Timeout(60.0, connect=10.0)  # 60s total, 10s pour la connexion
_http_client = httpx.Client(timeout=_timeout, limits=httpx.Limits(max_connections=10, max_keepalive_connections=5))

client = OpenAI(
    api_key=settings.together_ai_API_KEY,
    base_url=settings.together_ai_BASE_URL,
    http_client=_http_client,
    max_retries=2
)

MODEL = settings.together_ai_MODEL


def _clean_null_strings(obj):
    """
    Recursively clean 'null' strings and convert them to None.
    This fixes cases where LLM returns the string 'null' instead of null value.
    """
    if isinstance(obj, dict):
        return {k: _clean_null_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_null_strings(item) for item in obj]
    elif isinstance(obj, str):
        # Convert string 'null' to None
        if obj.lower() == 'null':
            return None
        return obj
    return obj


def safe_json_parse(raw_response: str, fallback: dict = None) -> dict:
    """
    Cleans and safely parses JSON returned by LLM.
    - Removes markdown or extra text around JSON.
    - Handles incomplete JSON by trying to repair it.
    - Converts string 'null' values to None.
    - Returns fallback dict if parsing fails.
    """
    if fallback is None:
        fallback = {}

    try:
        # Remove markdown code blocks if present
        cleaned = re.sub(r'^```json\s*|\s*```$', '', raw_response, flags=re.MULTILINE).strip()
        cleaned = re.sub(r'^```\s*|\s*```$', '', cleaned, flags=re.MULTILINE).strip()
        
        # Extract JSON content if wrapped in other text
        # First try to find complete JSON (with closing brace)
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            cleaned = match.group(0)
        else:
            # If no closing brace found, try to extract from first opening brace to end
            # This handles incomplete JSON that was truncated
            match_start = cleaned.find('{')
            if match_start >= 0:
                cleaned = cleaned[match_start:]
            else:
                cleaned = cleaned.strip()

        # Parse JSON
        parsed = json.loads(cleaned)
        # Clean 'null' strings recursively
        cleaned_parsed = _clean_null_strings(parsed)
        return cleaned_parsed
    except json.JSONDecodeError as e:
        # Try to repair incomplete JSON by closing brackets/braces
        try:
            # Remove any trailing whitespace
            cleaned = cleaned.rstrip()
            
            # Remove trailing comma if present
            cleaned = re.sub(r',\s*$', '', cleaned)
            
            # Count open and close braces/brackets
            open_braces = cleaned.count('{')
            close_braces = cleaned.count('}')
            open_brackets = cleaned.count('[')
            close_brackets = cleaned.count(']')
            
            # Check if we're in the middle of a string (odd number of unescaped quotes)
            # This is a simple check - count quotes that aren't escaped
            quote_count = len(re.findall(r'(?<!\\)"', cleaned))
            in_string = (quote_count % 2) != 0
            
            # If we're in a string, try to close it
            if in_string and not cleaned.endswith('"'):
                # Find the last unclosed quote and close the string
                cleaned += '"'
            
            # Handle incomplete values
            # If ends with colon (incomplete key-value pair), add null
            if cleaned.rstrip().endswith(':'):
                cleaned = cleaned.rstrip() + ' null'
            # If ends with a trailing comma before closing, it's usually fine but remove if needed
            # (We already removed trailing comma above, but check again after string handling)
            cleaned = re.sub(r',\s*$', '', cleaned.rstrip())
            
            # Close brackets first (inner structures), then braces (outer structure)
            while open_brackets > close_brackets:
                cleaned += ']'
                close_brackets += 1
            
            # Close braces
            while open_braces > close_braces:
                cleaned += '}'
                close_braces += 1
            
            # Try parsing the repaired JSON
            repaired_result = json.loads(cleaned)
            # Clean 'null' strings recursively
            repaired_result = _clean_null_strings(repaired_result)
            print(f"[✓ JSON Repaired] Successfully repaired incomplete JSON")
            return repaired_result
        except Exception as repair_error:
            # If repair failed, try one more approach: extract the largest valid JSON substring
            try:
                # Try to find a valid JSON object by gradually removing from the end
                for i in range(len(cleaned), 0, -1):
                    try:
                        test_json = cleaned[:i].rstrip()
                        # Remove trailing comma
                        test_json = re.sub(r',\s*$', '', test_json)
                        # Try to close it properly
                        open_br = test_json.count('{') - test_json.count('}')
                        open_sq = test_json.count('[') - test_json.count(']')
                        test_json += ']' * open_sq + '}' * open_br
                        parsed = json.loads(test_json)
                        # Clean 'null' strings recursively
                        parsed = _clean_null_strings(parsed)
                        print(f"[✓ JSON Partially Repaired] Extracted valid JSON subset")
                        return parsed
                    except:
                        continue
            except:
                pass
        
        print(f"[⚠️ JSON Decode Error] {e}")
        print(f"Raw response was:\n{raw_response}")
        print(f"Cleaned response was:\n{cleaned}")
        return fallback


# ============================================================
# 📄 1. Extract CV Information
# ============================================================
def extract_cv_info_via_llm(text_cv: str):
    prompt = f"""
You are an intelligent assistant that extracts structured data from a CV text.

**Important rules:**
- Return ONLY a valid raw JSON (no text, no Markdown, no formatting).
- If information is missing, use empty string "", empty list [], or null.
- Translate all extracted values to English.
- For professionalExperience, try to calculate years of experience from dates if possible.
- Order professionalExperience entries in reverse chronological order (most recent first).
- Merge duplicate roles (same company + position + identical dates) instead of repeating them.
- If dates are missing, explicitly set `"dates": null` and mention the absence in the description.

Follow exactly this structure:

{{
    "fullName": "string",
    "education": [{{"school": "string", "degree": "string", "dates": "string|null", "EQFLevel": "string|null"}}],
    "professionalExperience": [{{"company": "string", "position": "string", "dates": "string|null", "description": "string|null", "yearsOfExperience": "float|null"}}],
    "hardSkills": ["string"],
    "softSkills": ["string"],
    "address": "string|null",
    "email": "string|null",
    "phone": "string|null",
    "languages": [{{"language": "string", "level": "string|null"}}],
    "drivingLicense": "string|null",
    "projects": [{{"title": "string", "description": "string|null", "dates": "string|null"}}],
    "technologies": ["string"]
}}

Here is the CV text:
\"\"\"{text_cv}\"\"\"
"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        raw_response = response.choices[0].message.content.strip()
        return safe_json_parse(raw_response)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError, ConnectionResetError) as e:
        print(f"[API Error] Connection error in extract_cv_info_via_llm: {e}")
        return {}
    except Exception as e:
        print(f"[API Error] Unexpected error in extract_cv_info_via_llm: {e}")
        return {}


# ============================================================
# 💼 2. Analyze CV for Experience
# ============================================================
def analyze_cv_for_experience(text_cv: str, cv_id: str):
    prompt = f"""
You are an intelligent analyst who evaluates CVs with a **strong focus on professional experience**.

Your mission is to build an objective, insight-driven synthesis in **clear English** using only the facts contained in the CV text below.

### ⚙️ Core Rules
- Return **ONLY** a valid **raw JSON object** — no markdown, comments, or extra text.
- If a value is missing, use "" (empty string), [] (empty list), or null.
- Translate all extracted fields into English while keeping domain terminology precise.
- **Never reuse or lightly rephrase the candidate's own summary or headline.** Construct your own expert summary derived from the detailed experience you extract.
- Always cross-check dates and responsibilities. Highlight any inconsistencies (overlapping dates, missing years) directly in the summary text.
- Keep professionalExperience narratives in strict reverse chronological order (latest role first).

### 🎯 Experience Analysis Requirements
- **totalYearsExperience**: Estimate cumulative professional experience in years (numeric). Use partial years when needed.
- **experienceLevel**: Infer from the experience span:
  - junior: 0-2 years
  - mid: 2-5 years
  - senior: 5+ years
- **experienceDetails**: Produce a rich narrative organised in reverse chronological order. For EACH role include:
  - Company and position
  - Rough duration (even if approximate)
  - Core missions, achievements, and measurable impact
  - Technologies, tools, or methodologies used
  - Context (industry, team size, project type) when available
- **summaryText**: Craft an analytical overview (6-8 sentences). Highlight seniority, scope, strongest competencies, industries, and notable outcomes. Mention gaps or uncertainties if detected.
- **hardSkills**: Extract every explicit technical skill, language, tool, or framework.
- **softSkills**: Capture behavioural strengths (leadership, communication, etc.) that are evident in the text.
- **education**: Summarize the highest or most recent education in one sentence (institution + degree + year when possible).

### 🧩 Expected JSON Structure

{{
  "cv_id": "{cv_id}",
  "hardSkills": ["string"],
  "softSkills": ["string"],
  "experienceLevel": "junior|mid|senior|null",
  "education": "string|null",
  "summaryText": "string",
  "totalYearsExperience": "float|null",
  "experienceDetails": "string"
}}

### 📄 CV Text
\"\"\"{text_cv}\"\"\"
"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        raw_response = response.choices[0].message.content.strip()
        return safe_json_parse(raw_response, fallback={"cv_id": cv_id})
    except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError, ConnectionResetError) as e:
        print(f"[API Error] Connection error in analyze_cv_for_experience: {e}")
        return {"cv_id": cv_id}
    except Exception as e:
        print(f"[API Error] Unexpected error in analyze_cv_for_experience: {e}")
        return {"cv_id": cv_id}


# ============================================================
# 📞 3. Extract Phone Number
# ============================================================
def extract_phone_number_from_text(cv_text: str) -> dict:
    prompt = f"""
You are an intelligent assistant that extracts phone numbers from CV text.

**Rules:**
- Return ONLY valid raw JSON
- If no phone found, use null

Structure:
{{
    "phone_number": "string|null"
}}

CV TEXT:
{cv_text}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw_response = response.choices[0].message.content.strip()
        return safe_json_parse(raw_response, fallback={"phone_number": None})
    except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError, ConnectionResetError) as e:
        print(f"[API Error] Connection error in extract_phone_number_from_text: {e}")
        return {"phone_number": None}
    except Exception as e:
        print(f"[API Error] Unexpected error in extract_phone_number_from_text: {e}")
        return {"phone_number": None}
