import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Configuration
TIEBREAKER_LOGGING_ENABLED = True  # Set to False to disable tiebreaker justifications and logging

languages = [
    {'nl': 'Dutch'},
    {'de': 'German'},
    {'es': 'Spanish'},
    {'pt': 'Portuguese'},
    {'fr': 'French'},
    {'it': 'Italian'},
    {'ja': 'Japanese'},
    {'ko': 'Korean'},
    {'zh': 'Chinese'}
]

context = """
**Contextual Information**:
- "Controller" always refers to a "temperature controller," such as a BrewPi.
- A "profile" generally refers to a "temperature control profile" - a schedule of temperatures that fermenting liquid should be held at for a period of time to ensure proper fermentation.
- "Gravity" always refers to "specific gravity" - the measure of the sugar content remaining in a fermenting liquid such as beer.
- References to "original gravity", "OG", or "O.G." mean the starting gravity of the liquid.
"""


def generate_schema_from_value(value):
    """
    Generate a JSON schema from a Python value structure.
    This allows Structured Outputs to guarantee the response matches the input structure.
    """
    if isinstance(value, str):
        return {"type": "string"}
    elif isinstance(value, dict):
        properties = {}
        for k, v in value.items():
            properties[k] = generate_schema_from_value(v)
        return {
            "type": "object",
            "properties": properties,
            "required": list(value.keys()),
            "additionalProperties": False
        }
    elif isinstance(value, list):
        if len(value) > 0:
            return {
                "type": "array",
                "items": generate_schema_from_value(value[0])
            }
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


def run_tiebreaker(original_english, existing_translation, new_translation, target_language, key_path):
    """
    Use gpt-5.2 to determine which translation is better when the source string hasn't changed.
    Returns a tuple of (preferred_translation, tiebreaker_record or None).
    """
    if TIEBREAKER_LOGGING_ENABLED:
        schema = {
            "type": "object",
            "properties": {
                "preferred": {
                    "type": "string",
                    "enum": ["existing", "new"]
                },
                "justification": {
                    "type": "string"
                }
            },
            "required": ["preferred", "justification"],
            "additionalProperties": False
        }
    else:
        schema = {
            "type": "object",
            "properties": {
                "preferred": {
                    "type": "string",
                    "enum": ["existing", "new"]
                }
            },
            "required": ["preferred"],
            "additionalProperties": False
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

    if TIEBREAKER_LOGGING_ENABLED:
        system_content += "\n\nAlso provide a brief English-language justification for your choice."

    prompt = f"""Original English: "{original_english}"

Existing {target_language} translation: "{existing_translation}"

New {target_language} translation: "{new_translation}"

Which translation is better?"""

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "tiebreaker_result",
                "schema": schema,
                "strict": True
            }
        }
    )

    message = response.choices[0].message
    if hasattr(message, 'refusal') and message.refusal:
        # On refusal, prefer existing translation
        return existing_translation, None

    result = json.loads(message.content)
    preferred = existing_translation if result["preferred"] == "existing" else new_translation

    tiebreaker_record = None
    if TIEBREAKER_LOGGING_ENABLED:
        tiebreaker_record = {
            "key_path": key_path,
            "original_english": original_english,
            "existing_translation": existing_translation,
            "new_translation": new_translation,
            "preferred": result["preferred"],
            "justification": result.get("justification", "")
        }

    return preferred, tiebreaker_record


def get_nested_value(data, key_path):
    """Get a value from a nested dict using a dot-separated key path."""
    keys = key_path.split(".")
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def apply_tiebreakers(
    new_translation,
    existing_translation,
    source_english,
    translated_from_english,
    target_language,
    key_path_prefix=""
):
    """
    Recursively compare translations and apply tiebreakers where needed.

    Returns a tuple of (final_translation, list of tiebreaker_records).
    """
    tiebreaker_records = []

    if isinstance(new_translation, dict):
        result = {}
        for key, new_value in new_translation.items():
            current_path = f"{key_path_prefix}.{key}" if key_path_prefix else key
            existing_value = existing_translation.get(key) if isinstance(existing_translation, dict) else None
            source_value = source_english.get(key) if isinstance(source_english, dict) else None
            translated_from_value = translated_from_english.get(key) if isinstance(translated_from_english, dict) else None

            final_value, records = apply_tiebreakers(
                new_value,
                existing_value,
                source_value,
                translated_from_value,
                target_language,
                current_path
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
            preferred, record = run_tiebreaker(
                source_english,
                existing_translation,
                new_translation,
                target_language,
                key_path_prefix
            )
            if record:
                tiebreaker_records.append(record)
            return preferred, tiebreaker_records

        # If existing and new are the same, just return it
        return new_translation, []

    else:
        # For non-string, non-dict values, return as-is
        return new_translation, []


def translate_to_language(english_value, target_language="French", foreign_translation=None):
    """
    Submit the english_value to OpenAI for translation to target_language.
    Uses Structured Outputs to guarantee the response matches the expected schema.
    """
    # Generate schema from the input value structure
    value_schema = generate_schema_from_value(english_value)

    # Root must be an object for Structured Outputs, so wrap non-objects
    needs_unwrap = False
    if value_schema.get("type") != "object":
        needs_unwrap = True
        schema = {
            "type": "object",
            "properties": {"translation": value_schema},
            "required": ["translation"],
            "additionalProperties": False
        }
        input_data = {"translation": english_value}
        foreign_data = {"translation": foreign_translation} if foreign_translation else None
    else:
        schema = value_schema
        input_data = english_value
        foreign_data = foreign_translation

    english_text = json.dumps(input_data, ensure_ascii=False)
    prompt = f"Translate the following JSON containing English text to {target_language}:\n```\n{english_text}\n```\n"

    if foreign_data:
        foreign_text = json.dumps(foreign_data, ensure_ascii=False)
        prompt += f"\n\nHere is the existing translation in {target_language} for reference. The English text may have changed since this translation was last updated. If the translation is no longer correct or up-to-date please correct it:\n```\n{foreign_text}\n```\n"

    system_content = """You are a helpful assistant that translates English to other languages. The content to translate is provided as JSON. You provide the output as JSON matching the exact same structure.

Rules:
- Maintain all keys from the input exactly as they are
- Any values that begin with '@:' should remain unchanged (these are references)
- When you encounter values enclosed in braces like '{variable_name}', keep the variable name unchanged. The placeholder position can change to fit the target language grammar.
- Translate all user-facing text naturally for the target language

""" + context

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "translation_output",
                "schema": schema,
                "strict": True
            }
        }
    )

    # Handle refusals
    message = response.choices[0].message
    if hasattr(message, 'refusal') and message.refusal:
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


def translate_recursive(data, target_language, foreign_translation=None, translated_from=None):
    """
    Split nested directories at the first level of keys, translate, and apply tiebreakers.

    Returns a tuple of (translated_data, list of tiebreaker_records).
    """
    all_tiebreaker_records = []

    if isinstance(data, dict):
        output = {}
        for key, value in data.items():
            foreign_value = foreign_translation.get(key) if foreign_translation else None
            translated_from_value = translated_from.get(key) if translated_from else None
            try:
                # Get the new translation
                new_translation = translate_to_language(value, target_language, foreign_value)

                # Apply tiebreaker logic
                final_translation, records = apply_tiebreakers(
                    new_translation,
                    foreign_value,
                    value,
                    translated_from_value,
                    target_language,
                    key
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
        new_translation = translate_to_language(data, target_language, foreign_translation)
        final_translation, records = apply_tiebreakers(
            new_translation,
            foreign_translation,
            data,
            translated_from,
            target_language,
            ""
        )
        return final_translation, records
    else:
        return data, []


def translate_json_file(input_file, output_file, target_language="French"):
    with open(input_file, "r") as f:
        data = json.load(f)

    # Load existing translation if it exists
    data_translated = {}
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            data_translated = json.load(f)

    # Extract language codes from file names
    input_lang_code = os.path.basename(input_file).split('.')[0]
    output_lang_code = os.path.basename(output_file).split('.')[0]

    # Load the translated_from file if it exists
    translated_from_dir = os.path.join(os.path.dirname(output_file), "translated_from")
    cached_file_name = f"{output_lang_code}_{input_lang_code}.json"
    cached_file_path = os.path.join(translated_from_dir, cached_file_name)

    translated_from = {}
    if os.path.exists(cached_file_path):
        with open(cached_file_path, "r") as f:
            translated_from = json.load(f)

    # Translate with tiebreaker logic
    translated_data, tiebreaker_records = translate_recursive(
        data, target_language, data_translated, translated_from
    )

    # Write the translated output
    with open(output_file, "w") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

    # Create the translated_from directory if it doesn't exist
    os.makedirs(translated_from_dir, exist_ok=True)

    # Write the cached version of the input file
    with open(cached_file_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Write tiebreaker log if enabled and there are records
    if TIEBREAKER_LOGGING_ENABLED and tiebreaker_records:
        tiebreaker_log_dir = os.path.join(os.path.dirname(output_file), "tiebreaker_logs")
        os.makedirs(tiebreaker_log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"{output_lang_code}_{timestamp}.json"
        log_file_path = os.path.join(tiebreaker_log_dir, log_file_name)

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "source_file": input_file,
            "target_file": output_file,
            "target_language": target_language,
            "tiebreakers": tiebreaker_records
        }

        with open(log_file_path, "w") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        print(f"  Tiebreaker log written to {log_file_path} ({len(tiebreaker_records)} decisions)")


if __name__ == "__main__":
    source_lang_code = "en"  # English

#     target_lang_code = "es"

    target_lang_codes = ["es", "de", "nl", "pt"]

    for target_lang_code in target_lang_codes:
        input_filepath = f"../src/locales/{source_lang_code}.json"
        output_filepath = f"../src/locales/{target_lang_code}.json"
        target_lang = next((lang[target_lang_code] for lang in languages if target_lang_code in lang), None)
        # Tell the user what we're doing
        print(f"Translating {input_filepath} to {output_filepath} in {target_lang}")
        if not target_lang:
            raise ValueError(f"Language code '{target_lang_code}' not found in the languages list.")
        translate_json_file(input_filepath, output_filepath, target_lang)
        print("...done")
