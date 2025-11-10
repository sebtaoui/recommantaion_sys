def data_individual(todo):
    return {
        "id": str(todo["_id"]),
        "fullName": todo.get("fullName", ""),
        "education": todo.get("education", []),
        "professionalExperience": todo.get("professionalExperience", []),
        "hardSkills": todo.get("hardSkills", []),
        "softSkills": todo.get("softSkills", []),
        "address": todo.get("address", ""),
        "email": todo.get("email", ""),
        "phone": todo.get("phone", ""),
        "languages": todo.get("languages", []),
        "drivingLicense": todo.get("drivingLicense", ""),
        "projects": todo.get("projects", []),
        "technologies": todo.get("technologies", []),
    }

def all_tasks(todos):
    return [data_individual(todo) for todo in todos]

def data_individual_job_search(todo):
    return {
        "id": str(todo["_id"]),
        "cv_id": str(todo.get("cv_id", "")),
        "hardSkills": todo.get("hardSkills", []),
        "softSkills": todo.get("softSkills", []),
        "experienceLevel": todo.get("experienceLevel", None),
        "education": todo.get("education", None),
        "summaryText": todo.get("summaryText", ""),
        "totalYearsExperience": todo.get("totalYearsExperience", None),
        "experienceDetails": todo.get("experienceDetails", "")
    }

def all_tasks_search(todos):
    return [data_individual_job_search(todo) for todo in todos]
