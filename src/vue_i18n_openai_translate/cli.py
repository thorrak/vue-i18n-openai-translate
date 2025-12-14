"""Command-line interface for vue-i18n-openai-translate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from . import __version__
from .translator import translate_locale_directory


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    # Load environment variables from .env file in current working directory
    load_dotenv(find_dotenv(usecwd=True))

    parser = argparse.ArgumentParser(
        prog="vue-i18n-translate",
        description="Translate vue-i18n JSON locale files using OpenAI's API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect all locales in directory, translate from English
  vue-i18n-translate ./src/locales

  # Specify base locale and targets
  vue-i18n-translate ./src/locales -b en -t de -t es -t fr

  # With custom context file
  vue-i18n-translate ./src/locales -c ./translation-context.json

  # Dry run to see what would be translated
  vue-i18n-translate ./src/locales --dry-run

Environment Variables:
  OPENAI_API_KEY    Your OpenAI API key (required)
        """,
    )

    parser.add_argument(
        "locales_dir",
        type=Path,
        help="Path to the locales directory containing JSON files",
    )

    parser.add_argument(
        "-b",
        "--base-locale",
        type=str,
        default="en",
        help="Source locale code (default: en)",
    )

    parser.add_argument(
        "-t",
        "--target-locale",
        type=str,
        action="append",
        dest="target_locales",
        metavar="CODE",
        help="Target locale code (can be specified multiple times). "
        "If not specified, auto-detects from existing files in the directory.",
    )

    parser.add_argument(
        "-c",
        "--context-file",
        type=Path,
        default=None,
        help="Path to JSON file containing translation context and glossary",
    )

    parser.add_argument(
        "--no-tiebreaker-log",
        action="store_true",
        help="Disable tiebreaker decision logging",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be translated without making API calls",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args(argv)

    # Validate locales directory
    if not args.locales_dir.is_dir():
        print(f"Error: Directory does not exist: {args.locales_dir}", file=sys.stderr)
        return 1

    # Determine context file: use explicit argument, or default to translation-context.json in locales_dir
    context_file = args.context_file
    if context_file is None:
        default_context = args.locales_dir / "translation-context.json"
        if default_context.exists():
            context_file = default_context
            print(f"Using default context file: {context_file}")

    # Validate context file if provided
    if context_file and not context_file.exists():
        print(f"Error: Context file not found: {context_file}", file=sys.stderr)
        return 1

    try:
        translate_locale_directory(
            locales_dir=args.locales_dir,
            base_locale=args.base_locale,
            target_locales=args.target_locales,
            context_file=context_file,
            enable_tiebreaker_logging=not args.no_tiebreaker_log,
            dry_run=args.dry_run,
        )
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
