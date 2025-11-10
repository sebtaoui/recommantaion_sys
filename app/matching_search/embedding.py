def build_text_from_json(doc, experience_weight=3.0):
    """
    Convert CV JSON to text for embeddings with HEAVY focus on professional experience.
    
    Args:
        doc: The CV document dictionary
        experience_weight: How many times to repeat experience text (default 3x)
    
    Returns:
        String representation optimized for experience-focused matching
    """
    parts = []

    def flatten(value):
        if isinstance(value, list):
            flat = []
            for item in value:
                if isinstance(item, dict):
                    flat.append(" ".join(str(v) for k, v in item.items() if v))
                else:
                    flat.append(str(item))
            return " ".join(flat)
        elif isinstance(value, dict):
            return " ".join(str(v) for v in value.values() if v)
        return str(value)

    if "experienceDetails" in doc and doc["experienceDetails"]:
        experience_text = flatten(doc["experienceDetails"])
        for _ in range(int(experience_weight)):
            parts.append(experience_text)
    
    if "totalYearsExperience" in doc and doc["totalYearsExperience"]:
        years_text = f"{doc['totalYearsExperience']} years of experience"
        parts.append(years_text)
    
    for key in ["experienceLevel", "hardSkills", "summaryText"]:
        if key in doc and doc[key]:
            parts.append(flatten(doc[key]))
    
    if "education" in doc and doc["education"]:
        parts.append(flatten(doc["education"]))
    
    if "softSkills" in doc and doc["softSkills"]:
        parts.append(flatten(doc["softSkills"]))

    return " | ".join(parts)
