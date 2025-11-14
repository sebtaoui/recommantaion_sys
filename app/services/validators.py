from __future__ import annotations

import re
from typing import List, Tuple

from app.database.models import CVInfo, ProfessionalExperience, CVSummary

YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()


def _parse_years(dates: str | None) -> Tuple[int | None, int | None]:
    if not dates:
        return None, None
    matches = YEAR_PATTERN.findall(dates)
    if not matches:
        matches = re.findall(r"(19|20)\d{2}", dates)
    numbers = re.findall(r"(19|20)\d{2}", dates)
    years = [int(match) for match in re.findall(r"(19|20)\d{2}", dates)]
    if len(years) == 0:
        return None, None
    if len(years) == 1:
        return years[0], None
    return min(years), max(years)


def _deduplicate_experiences(
    experiences: List[ProfessionalExperience],
) -> Tuple[List[ProfessionalExperience], List[str]]:
    deduped: List[ProfessionalExperience] = []
    seen = set()
    warnings: List[str] = []

    for exp in experiences:
        key = (
            _normalize(exp.company),
            _normalize(exp.position),
            _normalize(exp.dates),
        )
        if key in seen and any(key):
            warnings.append(
                f"Duplicate professional experience detected for company='{exp.company}' position='{exp.position}'."
            )
            continue
        seen.add(key)
        deduped.append(exp)

    return deduped, warnings


def _chronology_warnings(experiences: List[ProfessionalExperience]) -> List[str]:
    dated_experiences = []
    warnings: List[str] = []

    for exp in experiences:
        start_year, end_year = _parse_years(exp.dates or "")
        dated_experiences.append((start_year, end_year, exp))
        if start_year and end_year and end_year < start_year:
            warnings.append(
                f"Experience '{exp.position}' at '{exp.company}' has end year ({end_year}) earlier than start year ({start_year})."
            )

    dated_experiences = [
        item for item in dated_experiences if item[0] is not None or item[1] is not None
    ]
    ordered = sorted(
        dated_experiences,
        key=lambda item: (
            item[0] if item[0] is not None else 9999,
            item[1] if item[1] is not None else 9999,
        ),
        reverse=True,
    )

    if ordered and ordered != dated_experiences:
        warnings.append("Professional experiences were not provided in reverse chronological order.")

    return warnings


def post_process_cv_info(cv_info: CVInfo) -> Tuple[CVInfo, List[str]]:
    warnings: List[str] = []
    experiences, duplicate_warnings = _deduplicate_experiences(cv_info.professionalExperience or [])
    warnings.extend(duplicate_warnings)
    warnings.extend(_chronology_warnings(experiences))

    updated = cv_info.model_copy(update={"professionalExperience": experiences})

    return updated, warnings


def post_process_cv_summary(cv_summary: CVSummary) -> Tuple[CVSummary, List[str]]:
    warnings: List[str] = []
    total_years = cv_summary.totalYearsExperience
    if total_years is not None and total_years < 0:
        warnings.append("Total years of experience was negative and set to 0.")
        total_years = 0.0

    normalized_summary = cv_summary.model_copy(update={"totalYearsExperience": total_years})

    if cv_summary.summaryText:
        sentences = [
            fragment.strip().lower()
            for fragment in re.split(r"[.;!\n]+", cv_summary.summaryText)
            if len(fragment.strip()) > 3
        ]
        seen = set()
        duplicated_fragments = set()
        for fragment in sentences:
            if fragment in seen:
                duplicated_fragments.add(fragment)
            else:
                seen.add(fragment)
        if duplicated_fragments:
            warnings.append(
                "Summary text contains repeated sentences; please review for duplication."
            )

    return normalized_summary, warnings


