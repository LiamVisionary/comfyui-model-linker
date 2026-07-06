"""
Hardware-aware model preferences for Model Linker.

The base Model Linker answers "what local file looks most like the workflow's
filename?" This layer answers the adjacent release question: "if the workflow was
authored for another GPU stack, what local/downloadable artifact is a better fit
for this machine?"
"""

from __future__ import annotations

import json
import os
import platform
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

METADATA_FILE = Path(__file__).resolve().parents[1] / "metadata" / "hardware-preferences.json"


def current_hardware_profile() -> str:
    forced = os.environ.get("ZIMG_ACCELERATOR_PROFILE", "").strip().lower()
    if forced:
        return forced

    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "apple-silicon"

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "version") and getattr(torch.version, "hip", None):
            return "rocm"
    except Exception:
        pass

    if system == "darwin":
        return "apple-intel"
    return "cpu"


def _load_rules() -> Dict[str, Any]:
    try:
        return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"profiles": {}, "recommendations": []}


def _text_for_model(model: Dict[str, Any]) -> str:
    return " ".join(
        str(model.get(key, ""))
        for key in ("filename", "relative_path", "path", "category")
    ).lower()


def _text_for_missing(missing: Dict[str, Any]) -> str:
    parts = [
        missing.get("original_path", ""),
        missing.get("category", ""),
        missing.get("node_type", ""),
    ]
    for ref in missing.get("all_node_refs", []) or []:
        if isinstance(ref, dict):
            parts.extend([ref.get("original_path", ""), ref.get("node_type", "")])
    return " ".join(str(part) for part in parts).lower()


def _token_present(text: str, token: str) -> bool:
    token = str(token or "").strip().lower()
    if not token:
        return False
    if token.startswith("re:"):
        try:
            return re.search(token[3:], text) is not None
        except re.error:
            return False
    return token in text


def _category_matches(rule: Dict[str, Any], category: str) -> bool:
    categories = [str(c).lower() for c in rule.get("categories", []) if c]
    if not categories:
        return True
    normalized = str(category or "unknown").lower()
    if normalized == "unknown":
        return "unknown" in categories
    return normalized in categories


def _profile_matches(rule: Dict[str, Any], profile: str) -> bool:
    profiles = [str(p).lower() for p in rule.get("profiles", []) if p]
    return not profiles or profile.lower() in profiles


def _missing_matches(rule: Dict[str, Any], missing: Dict[str, Any]) -> bool:
    text = _text_for_missing(missing)
    needles = rule.get("matches_any", [])
    if needles and not any(_token_present(text, token) for token in needles):
        return False
    exclusions = rule.get("unless_any", [])
    if exclusions and any(_token_present(text, token) for token in exclusions):
        return False
    return _category_matches(rule, str(missing.get("category", "unknown")))


def hardware_score_for_model(model: Dict[str, Any], profile: Optional[str] = None) -> tuple[int, List[str]]:
    profile = profile or current_hardware_profile()
    rules = _load_rules()
    profile_rules = rules.get("profiles", {}).get(profile, {})
    text = _text_for_model(model)
    score = 0
    reasons: List[str] = []

    for pref in profile_rules.get("preferred_terms", []):
        term = pref.get("term", "")
        if _token_present(text, term):
            points = int(pref.get("score", 0))
            score += points
            reason = pref.get("reason")
            if reason:
                reasons.append(str(reason))

    for avoid in profile_rules.get("avoid_terms", []):
        term = avoid.get("term", "")
        if _token_present(text, term):
            points = int(avoid.get("score", 0))
            score -= points
            reason = avoid.get("reason")
            if reason:
                reasons.append(str(reason))

    return score, reasons


def _recommendations_for_missing(
    missing: Dict[str, Any],
    available_models: Iterable[Dict[str, Any]],
    profile: str,
) -> List[Dict[str, Any]]:
    rules = _load_rules()
    recommendations: List[Dict[str, Any]] = []
    available = list(available_models or [])

    for rule in rules.get("recommendations", []):
        if not _profile_matches(rule, profile) or not _missing_matches(rule, missing):
            continue

        local_matches = []
        local_names = [str(name).lower() for name in rule.get("local_names", []) if name]
        for model in available:
            model_text = _text_for_model(model)
            if local_names and not any(name in model_text for name in local_names):
                continue
            if rule.get("category") and model.get("category") != rule.get("category"):
                continue
            local_matches.append(model)

        rec = {
            "id": rule.get("id", ""),
            "profile": profile,
            "label": rule.get("label", ""),
            "reason": rule.get("reason", ""),
            "category": rule.get("category", missing.get("category", "unknown")),
            "filename": rule.get("filename", ""),
            "local_matches": local_matches,
        }
        if rule.get("download"):
            rec["download"] = rule["download"]
        recommendations.append(rec)

    return recommendations


def apply_hardware_preferences(
    missing: Dict[str, Any],
    matches: List[Dict[str, Any]],
    available_models: Iterable[Dict[str, Any]],
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    profile = profile or current_hardware_profile()
    recommendations = _recommendations_for_missing(missing, available_models, profile)

    by_path = {
        os.path.normpath(str(match.get("model", {}).get("path", ""))): match
        for match in matches
        if match.get("model", {}).get("path")
    }

    for recommendation in recommendations:
        for model in recommendation.get("local_matches", []):
            path = os.path.normpath(str(model.get("path", "")))
            if not path:
                continue
            if path in by_path:
                existing = by_path[path]
                existing["hardware_recommended"] = True
                existing["hardware_recommendation_id"] = recommendation.get("id", "")
                existing["hardware_reasons"] = [
                    reason for reason in [
                        recommendation.get("reason", ""),
                        *existing.get("hardware_reasons", []),
                    ] if reason
                ]
                continue
            match = {
                "model": model,
                "filename": model.get("filename", ""),
                "similarity": 0.92,
                "confidence": 92.0,
                "hardware_recommended": True,
                "hardware_recommendation_id": recommendation.get("id", ""),
                "hardware_reasons": [recommendation.get("reason", "")],
            }
            matches.append(match)
            by_path[path] = match

    for match in matches:
        model = match.get("model", {})
        score, reasons = hardware_score_for_model(model, profile)
        if match.get("hardware_recommended"):
            score += 40
        target_category = str(missing.get("category", "unknown") or "unknown").lower()
        match_category = str(model.get("category", "unknown") or "unknown").lower()
        category_match = target_category == "unknown" or target_category == match_category
        match["hardware_profile"] = profile
        match["hardware_score"] = score
        match["category_match"] = category_match
        match["hardware_reasons"] = [
            reason for reason in [*match.get("hardware_reasons", []), *reasons] if reason
        ][:4]
        match["hardware_preferred"] = score > 0
        confidence = float(match.get("confidence", 0) or 0)
        match["effective_confidence"] = round(confidence + max(min(score, 18), -18), 1)

    matches.sort(
        key=lambda item: (
            float(item.get("confidence", 0) or 0) >= 100,
            bool(item.get("hardware_recommended")),
            bool(item.get("category_match")),
            float(item.get("effective_confidence", item.get("confidence", 0)) or 0),
            float(item.get("confidence", 0) or 0),
        ),
        reverse=True,
    )

    return {
        "hardware_profile": profile,
        "hardware_recommendations": recommendations,
        "matches": matches,
    }


def preferred_download_source(missing: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for recommendation in missing.get("hardware_recommendations", []) or []:
        download = recommendation.get("download")
        if not download:
            continue
        source = dict(download)
        source.setdefault("filename", recommendation.get("filename") or source.get("filename", ""))
        source.setdefault("directory", recommendation.get("category") or missing.get("category", "checkpoints"))
        source.setdefault("match_type", "hardware-preferred")
        source.setdefault("confidence", 100)
        source["hardware_profile"] = recommendation.get("profile", current_hardware_profile())
        source["hardware_reason"] = recommendation.get("reason", "")
        return source
    return None
