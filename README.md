# JSON Locale Translator

A Python script that translates JSON locale files from English to other languages using OpenAI's API with Structured Outputs.

## Features

- Translates nested JSON locale files while preserving structure
- Uses OpenAI Structured Outputs to guarantee valid JSON responses
- Preserves placeholder variables (e.g., `{variable_name}`) and reference strings (e.g., `@:key`)
- Supports incremental translation with existing translations as reference

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

Edit the `target_lang_codes` list in `translate.py` to specify which languages to translate to, then run:

```bash
python translate.py
```

The script expects an English source file at `../src/locales/en.json` and will output translated files to the same directory.