"""Entry point for the Bubble Sheet Auto-Mark application.

For local development, run via briefcase or after installing the package:

    uv pip install -e .
    uv run bubble-mark

Or using briefcase:

    briefcase dev
"""
import os
import sys

# Add src to path so bubble_mark package is importable without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from bubble_mark.ui.app import main
except ModuleNotFoundError as exc:
    sys.exit(
        f"Error: {exc}\n\n"
        "Run inside the project virtual environment:\n"
        "  uv pip install -e . && uv run bubble-mark\n"
        "or use Briefcase:\n"
        "  briefcase dev"
    )

if __name__ == "__main__":
    main().main_loop()
