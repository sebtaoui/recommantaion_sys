from pydantic import BaseModel, EmailStr, field_validator
from typing import List, Optional
import re

# ----------------------------
# 🔹 Nested Models
# ----------------------------
class Education(BaseModel):
    school: Optional[str] = None
    degree: Optional[str] = None
    dates: Optional[str] = None
    EQFLevel: Optional[str] = None


class Language(BaseModel):
    language: Optional[str] = None
    level: Optional[str] = None


class Project(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    dates: Optional[str] = None


class ProfessionalExperience(BaseModel):
    company: Optional[str] = None
    position: Optional[str] = None
    dates: Optional[str] = None
    description: Optional[str] = None
    yearsOfExperience: Optional[float] = None


# ----------------------------
# 🔹 CVInfo (Main Model)
# ----------------------------
class CVInfo(BaseModel):
    fullName: Optional[str] = None
    education: List[Education] = []
    professionalExperience: List[ProfessionalExperience] = []
    hardSkills: List[str] = []
    softSkills: List[str] = []
    address: Optional[str] = None
    email: Optional[str] = None  # Changed from EmailStr → str for safety
    phone: Optional[str] = None
    languages: List[Language] = []
    drivingLicense: Optional[str] = None
    projects: List[Project] = []
    technologies: List[str] = []

    # ✅ Custom email validator (fixes your error)
    @field_validator("email", mode="before")
    def validate_email(cls, value):
        if not value:
            return None
        value = value.strip()
        # Try to fix missing '@' between words (e.g. "john doe.com" → "john@doe.com")
        if "@" not in value and " " in value:
            parts = value.split()
            if len(parts) == 2:
                value = f"{parts[0]}@{parts[1]}"
        # Final validation
        if not re.match(r"[^@]+@[^@]+\.[^@]+", value):
            return None
        return value


# ----------------------------
# 🔹 CV Summary (Experience Analysis)
# ----------------------------
class CVSummary(BaseModel):
    cv_id: Optional[str] = None
    hardSkills: List[str] = []
    softSkills: List[str] = []
    experienceLevel: Optional[str] = None
    education: Optional[str] = None
    summaryText: Optional[str] = None
    totalYearsExperience: Optional[float] = None
    experienceDetails: Optional[str] = None


# ----------------------------
# 🔹 Extracted Phone Number
# ----------------------------
class CVPhone(BaseModel):
    filename: str
    phone_number: Optional[str] = None
