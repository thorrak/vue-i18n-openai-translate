"""Core translation logic for vue-i18n-openai-translate."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .utils import detect_target_locales, get_language_name, load_context

# Load environment variables from .env file
load_dotenv()


def _get_client() -> OpenAI:
    """Get OpenAI client, initializing if needed."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it in your environment or in a .env file."
        )
    return OpenAI(api_key=api_key)


def _generate_schema_from_value(value):
    """
    Generate a JSON schema from a Python value structure.

    This allows Structured Outputs to guarantee the response matches the input structure.
    """
    if isinstance(value, str):
        return {"type": "string"}
    elif isinstance(value, dict):
        properties = {}
        for k, v in value.items():
            properties[k] = _generate_schema_from_value(v)
        return {
            "type": "object",
            "properties": properties,
            "required": list(value.keys()),
            "additionalProperties": False,
        }
    elif isinstance(value, list):
        if len(value) > 0:
            return {"type": "array", "items": _generate_schema_from_value(value[0])}
        return {"type": "array", "items": {"type": "string"}}
    elif isinstance(value, bool):
        return {"type": "boolean"}
    elif isinstance(value, int):
        return {"type": "integer"}
    elif isinstance(value, float):
        return {"type": "number"}
    elif value is None:
        return {"type": ["string", "null"]}
    else:
        return {"type": "string"}


def _run_tiebreaker(
    client: OpenAI,
    original_english: str,
    existing_translation: str,
    new_translation: str,
    target_language: str,
    key_path: str,
    context: str,
    enable_logging: bool,
) -> tuple[str, dict | None]:
    """
    Use GPT-4.1 to determine which translation is better when the source string hasn't changed.

    Returns a tuple of (preferred_translation, tiebreaker_record or None).
    """
    if enable_logging:
        schema = {
            "type": "object",
            "properties": {
                "preferred": {"type": "string", "enum": ["existing", "new"]},
                "justification": {"type": "string"},
            },
            "required": ["preferred", "justification"],
            "additionalProperties": False,
        }
    else:
        schema = {
            "type": "object",
            "properties": {
                "preferred": {"type": "string", "enum": ["existing", "new"]}
            },
            "required": ["preferred"],
            "additionalProperties": False,
        }

    system_content = f"""You are an expert translator evaluating translation quality. You will be given:
1. An original English string
2. An existing translation in {target_language}
3. A new translation in {target_language}

Your task is to determine which translation is better. Consider:
- Accuracy of meaning
- Natural phrasing in the target language
- Consistency with UI/UX conventions
- Proper handling of placeholders (text in curly braces like {{variable}})

{context}

Respond with "existing" if the existing translation is better or equivalent, or "new" if the new translation is clearly better."""

    if enable_logging:
        system_content += (
            "\n\nAlso provide a brief English-language justification for your choice."
        )

    prompt = f"""Original English: "{original_english}"

Existing {target_language} translation: "{existing_translation}"

New {target_language} translation: "{new_translation}"

Which translation is better?"""

    # TODO - Make the tiebreak model (currently "gpt-5.2") configurable from the .env file
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "tiebreaker_result", "schema": schema, "strict": True},
        },
    )

    message = response.choices[0].message
    if hasattr(message, "refusal") and message.refusal:
        return existing_translation, None

    result = json.loads(message.content)
    preferred = (
        existing_translation if result["preferred"] == "existing" else new_translation
    )

    tiebreaker_record = None
    if enable_logging:
        tiebreaker_record = {
            "key_path": key_path,
            "original_english": original_english,
            "existing_translation": existing_translation,
            "new_translation": new_translation,
            "preferred": result["preferred"],
            "justification": result.get("justification", ""),
        }

    return preferred, tiebreaker_record


def _apply_tiebreakers(
    client: OpenAI,
    new_translation,
    existing_translation,
    source_english,
    translated_from_english,
    target_language: str,
    context: str,
    enable_logging: bool,
    key_path_prefix: str = "",
) -> tuple:
    """
    Recursively compare translations and apply tiebreakers where needed.

    Returns a tuple of (final_translation, list of tiebreaker_records).
    """
    tiebreaker_records = []

    if isinstance(new_translation, dict):
        result = {}
        for key, new_value in new_translation.items():
            current_path = f"{key_path_prefix}.{key}" if key_path_prefix else key
            existing_value = (
                existing_translation.get(key)
                if isinstance(existing_translation, dict)
                else None
            )
            source_value = (
                source_english.get(key) if isinstance(source_english, dict) else None
            )
            translated_from_value = (
                translated_from_english.get(key)
                if isinstance(translated_from_english, dict)
                else None
            )

            final_value, records = _apply_tiebreakers(
                client,
                new_value,
                existing_value,
                source_value,
                translated_from_value,
                target_language,
                context,
                enable_logging,
                current_path,
            )
            result[key] = final_value
            tiebreaker_records.extend(records)
        return result, tiebreaker_records

    elif isinstance(new_translation, str):
        # This is a leaf string - apply the decision logic

        # Skip reference strings (they shouldn't be translated anyway)
        if new_translation.startswith("@:"):
            return new_translation, []

        # Case 1: New string (not in translated_from)
        if translated_from_english is None:
            return new_translation, []

        # Case 2: Changed string (source doesn't match translated_from)
        if source_english != translated_from_english:
            return new_translation, []

        # Case 3: Unchanged string - run tiebreaker
        if existing_translation is not None and existing_translation != new_translation:
            preferred, record = _run_tiebreaker(
                client,
                source_english,
                existing_translation,
                new_translation,
                target_language,
                key_path_prefix,
                context,
                enable_logging,
            )
            if record:
                tiebreaker_records.append(record)
            return preferred, tiebreaker_records

        # If existing and new are the same, just return it
        return new_translation, []

    else:
        # For non-string, non-dict values, return as-is
        return new_translation, []


def _translate_to_language(
    client: OpenAI,
    english_value,
    target_language: str,
    context: str,
    foreign_translation=None,
):
    """
    Submit the english_value to OpenAI for translation to target_language.

    Uses Structured Outputs to guarantee the response matches the expected schema.
    """
    # Generate schema from the input value structure
    value_schema = _generate_schema_from_value(english_value)

    # Root must be an object for Structured Outputs, so wrap non-objects
    needs_unwrap = False
    if value_schema.get("type") != "object":
        needs_unwrap = True
        schema = {
            "type": "object",
            "properties": {"translation": value_schema},
            "required": ["translation"],
            "additionalProperties": False,
        }
        input_data = {"translation": english_value}
        foreign_data = (
            {"translation": foreign_translation} if foreign_translation else None
        )
    else:
        schema = value_schema
        input_data = english_value
        foreign_data = foreign_translation

    english_text = json.dumps(input_data, ensure_ascii=False)
    prompt = f"Translate the following JSON containing English text to {target_language}:\n```\n{english_text}\n```\n"

    if foreign_data:
        foreign_text = json.dumps(foreign_data, ensure_ascii=False)
        prompt += f"\n\nHere is the existing translation in {target_language} for reference. The English text may have changed since this translation was last updated. If the translation is no longer correct or up-to-date please correct it:\n```\n{foreign_text}\n```\n"

    system_content = f"""You are a helpful assistant that translates English to other languages. The content to translate is provided as JSON. You provide the output as JSON matching the exact same structure.

Rules:
- Maintain all keys from the input exactly as they are
- Any values that begin with '@:' should remain unchanged (these are references)
- When you encounter values enclosed in braces like '{{variable_name}}', keep the variable name unchanged. The placeholder position can change to fit the target language grammar.
- Translate all user-facing text naturally for the target language

{context}"""

    # TODO - Make the main translation model (currently "gpt-5-mini") configurable from the .env file
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "translation_output", "schema": schema, "strict": True},
        },
    )

    # Handle refusals
    message = response.choices[0].message
    if hasattr(message, "refusal") and message.refusal:
        raise ValueError(f"Model refused to translate: {message.refusal}")

    # Check for incomplete response
    if response.choices[0].finish_reason == "length":
        raise ValueError("Response was truncated due to length limit")

    translation = message.content

    # Parse and unwrap if needed
    parsed = json.loads(translation)
    if needs_unwrap:
        return parsed["translation"]
    return parsed


def _translate_recursive(
    client: OpenAI,
    data,
    target_language: str,
    context: str,
    enable_tiebreaker_logging: bool,
    foreign_translation=None,
    translated_from=None,
) -> tuple:
    """
    Split nested dictionaries at the first level of keys, translate, and apply tiebreakers.

    Returns a tuple of (translated_data, list of tiebreaker_records).
    """
    all_tiebreaker_records = []

    if isinstance(data, dict):
        output = {}
        for key, value in data.items():
            foreign_value = foreign_translation.get(key) if foreign_translation else None
            translated_from_value = (
                translated_from.get(key) if translated_from else None
            )
            try:
                # Get the new translation
                new_translation = _translate_to_language(
                    client, value, target_language, context, foreign_value
                )

                # Apply tiebreaker logic
                final_translation, records = _apply_tiebreakers(
                    client,
                    new_translation,
                    foreign_value,
                    value,
                    translated_from_value,
                    target_language,
                    context,
                    enable_tiebreaker_logging,
                    key,
                )
                output[key] = final_translation
                all_tiebreaker_records.extend(records)

            except json.JSONDecodeError as e:
                print(f"Unable to decode JSON for key '{key}': {e}")
                output[key] = value
            except ValueError as e:
                print(f"Translation error for key '{key}': {e}")
                output[key] = value
        return output, all_tiebreaker_records
    elif isinstance(data, str):
        new_translation = _translate_to_language(
            client, data, target_language, context, foreign_translation
        )
        final_translation, records = _apply_tiebreakers(
            client,
            new_translation,
            foreign_translation,
            data,
            translated_from,
            target_language,
            context,
            enable_tiebreaker_logging,
            "",
        )
        return final_translation, records
    else:
        return data, []


def translate_json_file(
    input_file: Path,
    output_file: Path,
    target_language: str,
    context: str = "",
    enable_tiebreaker_logging: bool = True,
) -> None:
    """
    Translate a JSON locale file from English to a target language.

    Args:
        input_file: Path to the source English JSON file
        output_file: Path to write the translated JSON file
        target_language: Full name of the target language (e.g., 'German')
        context: Optional context string for the translation
        enable_tiebreaker_logging: Whether to log tiebreaker decisions
    """
    client = _get_client()

    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    # Load existing translation if it exists
    data_translated = {}
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            data_translated = json.load(f)

    # Extract language codes from file names
    input_lang_code = input_file.stem
    output_lang_code = output_file.stem

    # Load the translated_from file if it exists
    translated_from_dir = output_file.parent / "translated_from"
    cached_file_name = f"{output_lang_code}_{input_lang_code}.json"
    cached_file_path = translated_from_dir / cached_file_name

    translated_from = {}
    if cached_file_path.exists():
        with open(cached_file_path, encoding="utf-8") as f:
            translated_from = json.load(f)

    # Translate with tiebreaker logic
    translated_data, tiebreaker_records = _translate_recursive(
        client,
        data,
        target_language,
        context,
        enable_tiebreaker_logging,
        data_translated,
        translated_from,
    )

    # Write the translated output
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

    # Create the translated_from directory if it doesn't exist
    translated_from_dir.mkdir(parents=True, exist_ok=True)

    # Write the cached version of the input file
    with open(cached_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Write tiebreaker log if enabled and there are records
    if enable_tiebreaker_logging and tiebreaker_records:
        tiebreaker_log_dir = output_file.parent / "tiebreaker_logs"
        tiebreaker_log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"{output_lang_code}_{timestamp}.json"
        log_file_path = tiebreaker_log_dir / log_file_name

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "source_file": str(input_file),
            "target_file": str(output_file),
            "target_language": target_language,
            "tiebreakers": tiebreaker_records,
        }

        with open(log_file_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        print(
            f"  Tiebreaker log written to {log_file_path} ({len(tiebreaker_records)} decisions)"
        )


def translate_locale_directory(
    locales_dir: Path,
    base_locale: str = "en",
    target_locales: list[str] | None = None,
    context_file: Path | None = None,
    enable_tiebreaker_logging: bool = True,
    dry_run: bool = False,
) -> None:
    """
    Translate all locale files in a directory.

    Args:
        locales_dir: Path to the locales directory
        base_locale: Source locale code (default: 'en')
        target_locales: List of target locale codes, or None to auto-detect
        context_file: Optional path to context JSON file
        enable_tiebreaker_logging: Whether to log tiebreaker decisions
        dry_run: If True, show what would be translated without making API calls
    """
    locales_dir = Path(locales_dir)

    if not locales_dir.is_dir():
        raise ValueError(f"Locales directory does not exist: {locales_dir}")

    # Load context
    context = load_context(context_file) if context_file else ""

    # Get source file
    source_file = locales_dir / f"{base_locale}.json"
    if not source_file.exists():
        raise FileNotFoundError(
            f"Base locale file not found: {source_file}"
        )

    # Detect or use provided target locales
    if target_locales is None:
        target_locales = detect_target_locales(locales_dir, base_locale)

    if not target_locales:
        print("No target locales found or specified. Nothing to translate.")
        return

    print(f"Base locale: {base_locale}")
    print(f"Target locales: {', '.join(target_locales)}")
    print(f"Source file: {source_file}")
    if context:
        print("Context: loaded")
    print()

    if dry_run:
        print("Dry run mode - no translations will be performed.")
        print("\nWould translate to:")
        for target_code in target_locales:
            try:
                target_language = get_language_name(target_code)
                output_file = locales_dir / f"{target_code}.json"
                exists = " (exists)" if output_file.exists() else " (new)"
                print(f"  - {target_code} ({target_language}){exists}")
            except ValueError as e:
                print(f"  - {target_code}: {e}")
        return

    # Translate each target locale
    for target_code in target_locales:
        try:
            target_language = get_language_name(target_code)
        except ValueError as e:
            print(f"Skipping {target_code}: {e}")
            continue

        output_file = locales_dir / f"{target_code}.json"
        print(f"Translating to {target_language} ({target_code})...")

        translate_json_file(
            source_file,
            output_file,
            target_language,
            context,
            enable_tiebreaker_logging,
        )

        print(f"  Written to {output_file}")
        print()
