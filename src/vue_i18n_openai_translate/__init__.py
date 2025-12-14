"""vue-i18n-openai-translate: Translate vue-i18n JSON locale files using OpenAI."""

__version__ = "0.1.0"

from .translator import translate_json_file, translate_locale_directory
from .utils import detect_target_locales, get_language_name, load_context

__all__ = [
    "__version__",
    "translate_json_file",
    "translate_locale_directory",
    "detect_target_locales",
    "get_language_name",
    "load_context",
]
