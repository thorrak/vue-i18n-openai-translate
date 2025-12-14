# Vue-i18n JSON Locale Translator

A Python script that translates JSON locale files targeted at programs built leveraging vue-i18n from English to other 
languages using OpenAI's API with Structured Outputs.

This implementation tracks the original strings that were translated, ensuring that as strings in the base locale change
the modified strings are re-translated. Unchanged strings are also resubmitted for translation to both provide context 
to the model, and provide an opportunity to "tiebreak" between multiple translation options if previous translations no longer seem appropriate given the updated context.

## Features

- Translates nested JSON locale files while preserving structure
- Uses OpenAI Structured Outputs to guarantee valid JSON responses
- Preserves placeholder variables (e.g., `{variable_name}`) and reference strings (e.g., `@:key`)
- Supports incremental translation with existing translations as reference
- Provides for a "tiebreaker" mechanism to choose between multiple translation options

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


## License

This project is licensed under Apache-2.0 License. See the [LICENSE](LICENSE) file for details.
