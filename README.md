# vue-i18n-openai-translate

A Python CLI tool that translates JSON locale files (targeted at vue-i18n) from English to other languages using OpenAI's API with Structured Outputs.

This implementation tracks the original strings that were translated, ensuring that as strings in the base locale change, the modified strings are re-translated. Unchanged strings are also resubmitted for translation to provide context to the model and offer an opportunity to "tiebreak" between multiple translation options if previous translations no longer seem appropriate.

## Features

- Translates nested JSON locale files while preserving structure
- Uses OpenAI Structured Outputs to guarantee valid JSON responses
- Preserves placeholder variables (e.g., `{variable_name}`) and reference strings (e.g., `@:key`)
- Supports incremental translation with existing translations as reference
- Auto-detects target locales from existing files in the directory
- Provides a "tiebreaker" mechanism to choose between multiple translation options
- Customizable translation context via JSON file

## Installation

### Using uv (recommended)

You can install directly from the GitHub repository using `uv`:

    # using HTTPS
    uv add git+https://github.com/thorrak/vue-i18n-openai-translate.git

    # or using SSH
    uv add git+ssh://git@github.com/thorrak/vue-i18n-openai-translate.git

Or, if the package is on PyPI, simply run:

```bash
uv add vue-i18n-openai-translate
```

### Using pip

Install directly from the GitHub repository using `pip`:

    # using HTTPS
    pip install git+https://github.com/thorrak/vue-i18n-openai-translate.git
    
    # or using SSH
    pip install git+ssh://git@github.com/thorrak/vue-i18n-openai-translate.git

Or, if the package is on PyPI, simply run:

```bash
pip install vue-i18n-openai-translate
```

### From source

From source (editable / development install):

    git clone https://github.com/thorrak/vue-i18n-openai-translate.git
    cd `vue-i18n-openai-translate`
    python -m pip install -e .


## Configuration

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=your-api-key-here
```

Or create a `.env` file in your working directory:

```
OPENAI_API_KEY=your-api-key-here
```

### Model Configuration

You can optionally customize which OpenAI models are used for translation:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_TRANSLATION_MODEL` | Model used for main translation tasks | `gpt-5-mini` |
| `OPENAI_TIEBREAKER_MODEL` | Model used for tiebreaker decisions | `gpt-5.2` |

Example `.env` file with all options:

```
OPENAI_API_KEY=your-api-key-here
OPENAI_TRANSLATION_MODEL=gpt-5-mini
OPENAI_TIEBREAKER_MODEL=gpt-5.2
```

## Usage

### Basic Usage

Auto-detect all locales in a directory and translate from English:

```bash
vue-i18n-translate ./src/locales
```

This will:
1. Find the base locale file (`en.json` by default)
2. Detect other locale files (e.g., `de.json`, `fr.json`, `es.json`)
3. Translate the English content to each detected language

### Specifying Target Locales

To translate to specific languages only:

```bash
vue-i18n-translate ./src/locales -t de -t es -t fr
```

### Using a Different Base Locale

```bash
vue-i18n-translate ./src/locales --base-locale fr
```

### Custom Context File

Provide domain-specific context to improve translation quality. If a file named `translation-context.json` exists in the locales directory, it will be loaded automatically:

```bash
# Automatically uses ./src/locales/translation-context.json if it exists
vue-i18n-translate ./src/locales

# Or specify a custom context file explicitly
vue-i18n-translate ./src/locales --context-file ./my-context.json
```

### Dry Run

See what would be translated without making any API calls:

```bash
vue-i18n-translate ./src/locales --dry-run
```

### All Options

```
usage: vue-i18n-translate [-h] [-b BASE_LOCALE] [-t CODE] [-c CONTEXT_FILE]
                          [--no-tiebreaker-log] [--dry-run] [-v]
                          locales_dir

Translate vue-i18n JSON locale files using OpenAI's API

positional arguments:
  locales_dir           Path to the locales directory containing JSON files

options:
  -h, --help            show this help message and exit
  -b, --base-locale     Source locale code (default: en)
  -t, --target-locale CODE
                        Target locale code (can be specified multiple times).
                        If not specified, auto-detects from existing files.
  -c, --context-file CONTEXT_FILE
                        Path to JSON file containing translation context
  --no-tiebreaker-log   Disable tiebreaker decision logging
  --dry-run             Show what would be translated without making API calls
  -v, --version         show program's version number and exit
```

## Context File Format

Create a JSON file with translation context to improve accuracy for domain-specific terminology:

```json
{
  "instructions": "This is a brewing/fermentation application. Use terminology appropriate for homebrewing and beer making.",
  "glossary": {
    "Controller": "temperature controller, such as a BrewPi",
    "profile": "temperature control profile - a schedule of temperatures",
    "Gravity": "specific gravity - the measure of sugar content in fermenting liquid",
    "OG": "original gravity - the starting gravity of the liquid"
  }
}
```

The `instructions` field provides general context, while `glossary` defines specific terms that should be translated consistently.

## How It Works

### Translation Process

1. The tool reads the base locale JSON file (e.g., `en.json`)
2. For each target locale, it:
   - Loads any existing translation
   - Loads the cached "translated from" file to detect changes
   - Sends chunks to OpenAI for translation
   - Applies tiebreaker logic for unchanged strings
   - Writes the result and updates the cache

### Tiebreaker Logic

For strings that haven't changed since the last translation:
- The tool generates a new translation
- If it differs from the existing translation, GPT-4.1 evaluates both
- The better translation is kept based on accuracy, naturalness, and consistency

By default, tiebreaker decisions are logged to JSON files in the `tiebreaker_logs/` subdirectory. These logs include the original string, both translation options, and the model's reasoning for each decision. To disable this logging:

```bash
vue-i18n-translate ./src/locales --no-tiebreaker-log
```

Note that the reasoning behind the decision is only generated when tiebreaker logging is enabled - disabling logging will reduce API usage at the expense of transparency.

### Directory Structure

After running, your locales directory will look like:

```
locales/
├── en.json                    # Base locale (source)
├── de.json                    # German translation
├── fr.json                    # French translation
├── translation-context.json   # Optional context file (auto-loaded if present)
├── translated_from/           # Cache of source strings
│   ├── de_en.json
│   └── fr_en.json
└── tiebreaker_logs/           # Optional decision logs
    ├── de_20241214_103045.json
    └── fr_20241214_103112.json
```

## Programmatic Usage

You can also use the library programmatically:

```python
from pathlib import Path
from vue_i18n_openai_translate import translate_locale_directory, translate_json_file

# Translate all locales in a directory
translate_locale_directory(
    locales_dir=Path("./src/locales"),
    base_locale="en",
    target_locales=["de", "fr", "es"],
    context_file=Path("./context.json"),
)

# Or translate a single file
translate_json_file(
    input_file=Path("./src/locales/en.json"),
    output_file=Path("./src/locales/de.json"),
    target_language="German",
    context="Domain-specific context here...",
)
```

## License

This project is licensed under Apache-2.0 License. See the [LICENSE](LICENSE) file for details.
