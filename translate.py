import json
import os

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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


def translate_recursive(data, target_language, foreign_translation=None):
    """Split nested directories at the first level of keys, and translate"""
    if isinstance(data, dict):
        output = {}
        for key, value in data.items():
            foreign_value = foreign_translation.get(key) if foreign_translation else None
            try:
                translated = translate_to_language(value, target_language, foreign_value)
                output[key] = translated
            except json.JSONDecodeError as e:
                print(f"Unable to decode JSON for key '{key}': {e}")
                output[key] = value
            except ValueError as e:
                print(f"Translation error for key '{key}': {e}")
                output[key] = value
        return output
    elif isinstance(data, str):
        return translate_to_language(data, target_language, foreign_translation)
    else:
        return data


def translate_json_file(input_file, output_file, target_language="French"):
    with open(input_file, "r") as f:
        data = json.load(f)

    data_translated = {}
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            data_translated = json.load(f)

    translated_data = translate_recursive(data, target_language, data_translated)

    with open(output_file, "w") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

    # Extract language codes from file names
    input_lang_code = os.path.basename(input_file).split('.')[0]
    output_lang_code = os.path.basename(output_file).split('.')[0]

    # Create the translated_from directory if it doesn't exist
    translated_from_dir = os.path.join(os.path.dirname(output_file), "translated_from")
    os.makedirs(translated_from_dir, exist_ok=True)

    # Write the cached version of the input file
    cached_file_name = f"{output_lang_code}_{input_lang_code}.json"
    cached_file_path = os.path.join(translated_from_dir, cached_file_name)
    with open(cached_file_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

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