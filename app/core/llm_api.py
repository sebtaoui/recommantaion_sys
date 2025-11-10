from app.core.config import settings
from openai import OpenAI
import json
import re

client = OpenAI(
    api_key=settings.together_ai_API_KEY,
    base_url=settings.together_ai_BASE_URL
)

MODEL = settings.together_ai_MODEL


def safe_json_parse(raw_response: str, fallback: dict = None) -> dict:
    """
    Cleans and safely parses JSON returned by LLM.
    - Removes markdown or extra text around JSON.
    - Handles incomplete JSON by trying to repair it.
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

        # Try to parse
        return json.loads(cleaned)
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
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    raw_response = response.choices[0].message.content.strip()
    return safe_json_parse(raw_response)


# ============================================================
# 💼 2. Analyze CV for Experience
# ============================================================
def analyze_cv_for_experience(text_cv: str, cv_id: str):
    prompt = f"""
You are an intelligent assistant specialized in analyzing CVs with a **strong focus on professional experience**.

Your goal is to extract key structured data and translate all extracted text into **clear English**.

### ⚙️ **Instructions**
- The input text below is a CV (resumé).
- Return **ONLY** a valid **raw JSON object** — no markdown, no comments, no text outside the JSON.
- If information is missing, use empty string "", empty list [], or null.
- Translate all extracted fields into **English**, keeping professional terms accurate.
- **FOCUS HEAVILY ON PROFESSIONAL EXPERIENCE** - this is the most important part.

### 🎓 **Special rules for experience analysis**
- **totalYearsExperience**: Calculate total years of professional work experience (numeric value).
- **experienceLevel**: Infer based on years of experience:
  - junior: 0-2 years
  - mid: 2-5 years
  - senior: 5+ years
- **experienceDetails**: Write a detailed summary of ALL professional experience, including:
  - All companies worked at
  - All positions held
  - Key technologies used in each role
  - Notable achievements and responsibilities
  - Duration at each company
- **summaryText**: Write a concise professional summary highlighting experience and expertise.
- **hardSkills**: List ALL technical skills, tools, frameworks, and technologies mentioned.
- **softSkills**: Include interpersonal or behavioral strengths.

### 🧩 **Expected JSON Structure**

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

### 📄 **Here is the CV text:**
\"\"\"{text_cv}\"\"\"
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    raw_response = response.choices[0].message.content.strip()
    return safe_json_parse(raw_response, fallback={"cv_id": cv_id})


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
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw_response = response.choices[0].message.content.strip()
    return safe_json_parse(raw_response, fallback={"phone_number": None})
