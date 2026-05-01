"""Entry point for the Bubble Sheet Auto-Mark application."""
import os
import sys

# Add src to path so bubble_mark package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from bubble_mark.ui.app import BubbleMarkApp

if __name__ == "__main__":
    BubbleMarkApp().run()
