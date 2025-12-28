"""Core translation logic for vue-i18n-openai-translate."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI, RateLimitError

from .utils import detect_target_locales, get_language_name, load_context

# Load environment variables from .env file in current working directory
load_dotenv(find_dotenv(usecwd=True))

# Model configuration - can be overridden via environment variables or .env file
TRANSLATION_MODEL = os.environ.get("OPENAI_TRANSLATION_MODEL", "gpt-5-mini")
TIEBREAKER_MODEL = os.environ.get("OPENAI_TIEBREAKER_MODEL", "gpt-5.2")

# Tiebreak mode configuration
# Valid values: "tiebreak" (default), "choose_existing", "choose_new"
TIEBREAK_MODE = os.environ.get("TIEBREAK_MODE", "tiebreak")

# Rate limiting configuration
MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "10"))
INITIAL_RETRY_DELAY = float(os.environ.get("INITIAL_RETRY_DELAY", "1.0"))
MAX_BACKOFF_RESETS = int(os.environ.get("MAX_BACKOFF_RESETS", "5"))

# Tiebreaker batch size - controls how many tiebreaker API calls run in parallel per language
TIEBREAKER_BATCH_SIZE = int(os.environ.get("TIEBREAKER_BATCH_SIZE", "10"))

# Backoff thresholds (in seconds)
MAX_DELAY = 64  # Reset backoff after reaching this delay
RESET_DELAY = 16  # Delay to reset to after hitting MAX_DELAY

# Global semaphore for limiting concurrent API calls (initialized lazily)
_api_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create the global semaphore for rate limiting."""
    global _api_semaphore
    if _api_semaphore is None:
        _api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    return _api_semaphore


async def _api_call_with_retry(api_call_func, *args, **kwargs):
    """
    Execute an API call with rate limiting and exponential backoff retry.

    Backoff pattern: 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 16 -> 32 -> 64 -> ...
    Resets to 16s after hitting 64s. Times out after MAX_BACKOFF_RESETS resets
    (approximately 10 minutes with default settings).

    Args:
        api_call_func: The async function to call
        *args, **kwargs: Arguments to pass to the function

    Returns:
        The result of the API call

    Raises:
        RateLimitError: If all retries are exhausted after MAX_BACKOFF_RESETS resets
    """
    semaphore = _get_semaphore()
    delay = INITIAL_RETRY_DELAY
    reset_count = 0
    attempt = 0

    while True:
        async with semaphore:
            try:
                return await api_call_func(*args, **kwargs)
            except RateLimitError as e:
                attempt += 1

                # Check if we've exceeded max resets
                if reset_count >= MAX_BACKOFF_RESETS:
                    print(f"Rate limit retry timeout after {reset_count} backoff resets (~{MAX_BACKOFF_RESETS * 112}s)")
                    raise

                # Extract retry-after header if available
                retry_after = getattr(e, "retry_after", None)
                if retry_after:
                    wait_time = float(retry_after)
                else:
                    wait_time = delay

                print(f"Rate limited, waiting {wait_time:.1f}s (attempt {attempt}, reset {reset_count}/{MAX_BACKOFF_RESETS})...")
                await asyncio.sleep(wait_time)

                # Update delay with exponential backoff and reset logic
                delay *= 2
                if delay > MAX_DELAY:
                    delay = RESET_DELAY
                    reset_count += 1


def _get_client() -> AsyncOpenAI:
    """Get async OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it in your environment or in a .env file."
        )
    return AsyncOpenAI(api_key=api_key)


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


async def _run_tiebreaker(
    client: AsyncOpenAI,
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

    response = await _api_call_with_retry(
        client.chat.completions.create,
        model=TIEBREAKER_MODEL,
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


def _collect_tiebreaker_candidates(
    new_translation,
    existing_translation,
    source_english,
    translated_from_english,
    key_path_prefix: str = "",
) -> list[tuple[str, str, str, str]]:
    """
    Recursively collect all tiebreaker candidates from the translation tree.

    Returns a list of tuples: (key_path, original_english, existing_translation, new_translation)
    Only collects candidates where:
    - The value is a string (leaf node)
    - Not a reference string (@:...)
    - The source hasn't changed (source_english == translated_from_english)
    - The translations differ (existing != new)
    """
    candidates = []

    if isinstance(new_translation, dict):
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

            nested_candidates = _collect_tiebreaker_candidates(
                new_value,
                existing_value,
                source_value,
                translated_from_value,
                current_path,
            )
            candidates.extend(nested_candidates)

    elif isinstance(new_translation, str):
        # Skip reference strings
        if new_translation.startswith("@:"):
            return []

        # Skip new strings (not in translated_from)
        if translated_from_english is None:
            return []

        # Skip changed strings (source doesn't match translated_from)
        if source_english != translated_from_english:
            return []

        # Unchanged string with different translations - this is a tiebreaker candidate
        if existing_translation is not None and existing_translation != new_translation:
            candidates.append((
                key_path_prefix,
                source_english,
                existing_translation,
                new_translation,
            ))

    return candidates


async def _run_tiebreaker_batch(
    client: AsyncOpenAI,
    candidates: list[tuple[str, str, str, str]],
    target_language: str,
    context: str,
    enable_logging: bool,
    batch_size: int,
) -> dict[str, tuple[str, dict | None]]:
    """
    Execute tiebreaker API calls in parallel batches.

    Args:
        candidates: List of (key_path, original_english, existing_translation, new_translation)
        batch_size: Number of concurrent API calls per batch

    Returns:
        Dict mapping key_path -> (preferred_translation, tiebreaker_record or None)
        On failure, uses existing translation (conservative).
    """
    results = {}

    # Process in batches
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]

        # Create tasks for this batch
        tasks = [
            _run_tiebreaker(
                client,
                original_english,
                existing_translation,
                new_translation,
                target_language,
                key_path,
                context,
                enable_logging,
            )
            for key_path, original_english, existing_translation, new_translation in batch
        ]

        # Execute batch in parallel
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results, handling exceptions conservatively
        for (key_path, _, existing_translation, _), result in zip(batch, batch_results):
            if isinstance(result, Exception):
                # On error, use existing translation (conservative)
                print(f"Tiebreaker failed for '{key_path}': {result}. Using existing translation.")
                results[key_path] = (existing_translation, None)
            else:
                results[key_path] = result

    return results


def _apply_tiebreaker_results(
    new_translation,
    existing_translation,
    source_english,
    translated_from_english,
    tiebreaker_results: dict[str, tuple[str, dict | None]],
    tiebreak_mode: str = "tiebreak",
    key_path_prefix: str = "",
) -> tuple:
    """
    Recursively apply pre-computed tiebreaker results to build final translation.

    This is a pure function (no API calls) that uses the results from _run_tiebreaker_batch().

    Args:
        tiebreaker_results: Dict mapping key_path -> (preferred_translation, tiebreaker_record)
        tiebreak_mode: One of "tiebreak", "choose_existing", or "choose_new"

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

            final_value, records = _apply_tiebreaker_results(
                new_value,
                existing_value,
                source_value,
                translated_from_value,
                tiebreaker_results,
                tiebreak_mode,
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

        # Case 3: Unchanged string - apply tiebreak mode
        if existing_translation is not None and existing_translation != new_translation:
            if tiebreak_mode == "choose_existing":
                return existing_translation, []
            elif tiebreak_mode == "choose_new":
                return new_translation, []
            else:
                # Default "tiebreak" mode - look up pre-computed result
                if key_path_prefix in tiebreaker_results:
                    preferred, record = tiebreaker_results[key_path_prefix]
                    if record:
                        tiebreaker_records.append(record)
                    return preferred, tiebreaker_records
                else:
                    # No result found (shouldn't happen), default to existing (conservative)
                    return existing_translation, []

        # If existing and new are the same, just return it
        return new_translation, []

    else:
        # For non-string, non-dict values, return as-is
        return new_translation, []


async def _translate_to_language(
    client: AsyncOpenAI,
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

    response = await _api_call_with_retry(
        client.chat.completions.create,
        model=TRANSLATION_MODEL,
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


async def _translate_single_key(
    client: AsyncOpenAI,
    key: str,
    value,
    target_language: str,
    context: str,
    enable_tiebreaker_logging: bool,
    tiebreak_mode: str,
    foreign_value,
    translated_from_value,
    source_value,
) -> tuple[str, any, list]:
    """
    Translate a single top-level key and apply tiebreakers.

    Tiebreakers are collected and executed in parallel batches for improved performance.

    Returns a tuple of (key, final_translation, tiebreaker_records).
    """
    try:
        # Get the new translation
        new_translation = await _translate_to_language(
            client, value, target_language, context, foreign_value
        )

        # Collect tiebreaker candidates (no API calls, just tree traversal)
        candidates = _collect_tiebreaker_candidates(
            new_translation,
            foreign_value,
            source_value,
            translated_from_value,
            key,
        )

        # Execute tiebreakers in parallel batches if there are candidates and mode is "tiebreak"
        tiebreaker_results = {}
        if candidates and tiebreak_mode == "tiebreak":
            tiebreaker_results = await _run_tiebreaker_batch(
                client,
                candidates,
                target_language,
                context,
                enable_tiebreaker_logging,
                TIEBREAKER_BATCH_SIZE,
            )

        # Apply pre-computed tiebreaker results to build final translation
        final_translation, records = _apply_tiebreaker_results(
            new_translation,
            foreign_value,
            source_value,
            translated_from_value,
            tiebreaker_results,
            tiebreak_mode,
            key,
        )
        return key, final_translation, records

    except json.JSONDecodeError as e:
        print(f"Unable to decode JSON for key '{key}': {e}")
        return key, value, []
    except ValueError as e:
        print(f"Translation error for key '{key}': {e}")
        return key, value, []


async def _translate_recursive(
    client: AsyncOpenAI,
    data,
    target_language: str,
    context: str,
    enable_tiebreaker_logging: bool,
    tiebreak_mode: str = "tiebreak",
    foreign_translation=None,
    translated_from=None,
) -> tuple:
    """
    Split nested dictionaries at the first level of keys, translate, and apply tiebreakers.

    Translations of top-level keys are performed in parallel for improved performance.

    Returns a tuple of (translated_data, list of tiebreaker_records).
    """
    all_tiebreaker_records = []

    if isinstance(data, dict):
        # Build list of translation tasks for all top-level keys
        tasks = []
        for key, value in data.items():
            foreign_value = foreign_translation.get(key) if foreign_translation else None
            translated_from_value = (
                translated_from.get(key) if translated_from else None
            )
            task = _translate_single_key(
                client,
                key,
                value,
                target_language,
                context,
                enable_tiebreaker_logging,
                tiebreak_mode,
                foreign_value,
                translated_from_value,
                value,
            )
            tasks.append(task)

        # Run all key translations in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        output = {}
        for result in results:
            if isinstance(result, Exception):
                print(f"Translation task failed: {result}")
                continue
            key, final_translation, records = result
            output[key] = final_translation
            all_tiebreaker_records.extend(records)

        return output, all_tiebreaker_records
    elif isinstance(data, str):
        new_translation = await _translate_to_language(
            client, data, target_language, context, foreign_translation
        )

        # Collect tiebreaker candidates
        candidates = _collect_tiebreaker_candidates(
            new_translation,
            foreign_translation,
            data,
            translated_from,
            "",
        )

        # Execute tiebreakers in parallel batches if needed
        tiebreaker_results = {}
        if candidates and tiebreak_mode == "tiebreak":
            tiebreaker_results = await _run_tiebreaker_batch(
                client,
                candidates,
                target_language,
                context,
                enable_tiebreaker_logging,
                TIEBREAKER_BATCH_SIZE,
            )

        # Apply pre-computed results
        final_translation, records = _apply_tiebreaker_results(
            new_translation,
            foreign_translation,
            data,
            translated_from,
            tiebreaker_results,
            tiebreak_mode,
            "",
        )
        return final_translation, records
    else:
        return data, []


async def translate_json_file(
    client: AsyncOpenAI,
    input_file: Path,
    output_file: Path,
    target_language: str,
    context: str = "",
    enable_tiebreaker_logging: bool = True,
    tiebreak_mode: str = "tiebreak",
) -> dict:
    """
    Translate a JSON locale file from English to a target language.

    Args:
        client: AsyncOpenAI client instance
        input_file: Path to the source English JSON file
        output_file: Path to write the translated JSON file
        target_language: Full name of the target language (e.g., 'German')
        context: Optional context string for the translation
        enable_tiebreaker_logging: Whether to log tiebreaker decisions
        tiebreak_mode: One of "tiebreak", "choose_existing", or "choose_new"

    Returns:
        Dictionary with translation result info (target_language, output_file, tiebreaker_log_path)
    """
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
    translated_data, tiebreaker_records = await _translate_recursive(
        client,
        data,
        target_language,
        context,
        enable_tiebreaker_logging,
        tiebreak_mode,
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

    result = {
        "target_language": target_language,
        "output_file": output_file,
        "tiebreaker_log_path": None,
        "tiebreaker_count": 0,
    }

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

        result["tiebreaker_log_path"] = log_file_path
        result["tiebreaker_count"] = len(tiebreaker_records)

    return result


async def translate_locale_directory(
    locales_dir: Path,
    base_locale: str = "en",
    target_locales: list[str] | None = None,
    context_file: Path | None = None,
    enable_tiebreaker_logging: bool = True,
    tiebreak_mode: str | None = None,
    dry_run: bool = False,
) -> None:
    """
    Translate all locale files in a directory.

    Translations are performed in parallel using asyncio for improved performance.

    Args:
        locales_dir: Path to the locales directory
        base_locale: Source locale code (default: 'en')
        target_locales: List of target locale codes, or None to auto-detect
        context_file: Optional path to context JSON file
        enable_tiebreaker_logging: Whether to log tiebreaker decisions
        tiebreak_mode: One of "tiebreak", "choose_existing", or "choose_new".
            If None, uses TIEBREAK_MODE environment variable or defaults to "tiebreak"
        dry_run: If True, show what would be translated without making API calls
    """
    locales_dir = Path(locales_dir)

    if not locales_dir.is_dir():
        raise ValueError(f"Locales directory does not exist: {locales_dir}")

    # Load context
    context = load_context(context_file) if context_file else ""

    # Resolve tiebreak mode: use parameter, fall back to env var, then default
    resolved_tiebreak_mode = tiebreak_mode if tiebreak_mode is not None else TIEBREAK_MODE
    valid_modes = ("tiebreak", "choose_existing", "choose_new")
    if resolved_tiebreak_mode not in valid_modes:
        raise ValueError(
            f"Invalid tiebreak mode: '{resolved_tiebreak_mode}'. "
            f"Valid options are: {', '.join(valid_modes)}"
        )

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
    print(f"Tiebreak mode: {resolved_tiebreak_mode}")
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

    # Build list of translation tasks
    translation_tasks = []
    valid_locales = []

    # Get shared client for all translations
    client = _get_client()

    for target_code in target_locales:
        try:
            target_language = get_language_name(target_code)
        except ValueError as e:
            print(f"Skipping {target_code}: {e}")
            continue

        output_file = locales_dir / f"{target_code}.json"
        valid_locales.append((target_code, target_language))

        # Create coroutine for this translation
        task = translate_json_file(
            client,
            source_file,
            output_file,
            target_language,
            context,
            enable_tiebreaker_logging,
            resolved_tiebreak_mode,
        )
        translation_tasks.append(task)

    if not translation_tasks:
        print("No valid target locales to translate.")
        return

    # Print what we're translating
    print(f"Translating to {len(valid_locales)} languages in parallel...")
    for target_code, target_language in valid_locales:
        print(f"  - {target_language} ({target_code})")
    print()

    # Run all translations in parallel
    results = await asyncio.gather(*translation_tasks, return_exceptions=True)

    # Report results
    print("Translation results:")
    for (target_code, target_language), result in zip(valid_locales, results):
        if isinstance(result, Exception):
            print(f"  {target_language} ({target_code}): FAILED - {result}")
        else:
            print(f"  {target_language} ({target_code}): Written to {result['output_file']}")
            if result["tiebreaker_log_path"]:
                print(f"    Tiebreaker log: {result['tiebreaker_log_path']} ({result['tiebreaker_count']} decisions)")
    print()
