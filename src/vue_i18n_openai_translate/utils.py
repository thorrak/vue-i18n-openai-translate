"""Utility functions for vue-i18n-openai-translate."""

from __future__ import annotations

import json
from pathlib import Path

import pycountry


def get_language_name(code: str) -> str:
    """
    Get the full language name from an ISO 639-1 language code.

    Args:
        code: Two-letter ISO 639-1 language code (e.g., 'de', 'fr', 'es')

    Returns:
        Full language name (e.g., 'German', 'French', 'Spanish')

    Raises:
        ValueError: If the language code is not recognized
    """
    # Handle some common special cases
    special_cases = {
        "zh": "Chinese",
        "zh-cn": "Chinese (Simplified)",
        "zh-tw": "Chinese (Traditional)",
    }

    code_lower = code.lower()
    if code_lower in special_cases:
        return special_cases[code_lower]

    # Try to find the language using pycountry
    language = pycountry.languages.get(alpha_2=code_lower)
    if language:
        return language.name

    # Try alpha_3 code as fallback
    language = pycountry.languages.get(alpha_3=code_lower)
    if language:
        return language.name

    raise ValueError(f"Unknown language code: {code}")


def detect_target_locales(directory: Path, base_locale: str) -> list[str]:
    """
    Detect target locales from JSON files in a directory.

    Scans the directory for .json files and returns a list of locale codes,
    excluding the base locale, translation-context.json, and any files in subdirectories.

    Args:
        directory: Path to the locales directory
        base_locale: The base locale code to exclude (e.g., 'en')

    Returns:
        List of target locale codes found (e.g., ['de', 'fr', 'es'])
    """
    if not directory.is_dir():
        raise ValueError(f"Directory does not exist: {directory}")

    # Files to exclude from locale detection
    excluded_files = {base_locale, "translation-context"}

    locales = []
    for json_file in directory.glob("*.json"):
        locale_code = json_file.stem
        if locale_code not in excluded_files:
            locales.append(locale_code)

    return sorted(locales)


def load_context(path: Path | None) -> str:
    """
    Load translation context from a JSON file.

    The context file can have the following structure:
    {
        "instructions": "General instructions for the translator...",
        "glossary": {
            "term": "definition",
            ...
        }
    }

    Args:
        path: Path to the context JSON file, or None for no context

    Returns:
        Formatted context string for the translation prompt
    """
    if path is None:
        return ""

    if not path.exists():
        raise FileNotFoundError(f"Context file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    parts = []

    if "instructions" in data:
        parts.append("**Contextual Information**:")
        parts.append(data["instructions"])

    if "glossary" in data and isinstance(data["glossary"], dict):
        parts.append("\n**Glossary**:")
        for term, definition in data["glossary"].items():
            parts.append(f'- "{term}" refers to {definition}')

    return "\n".join(parts)
