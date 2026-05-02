"""Allow the package to be run with ``python -m bubble_mark`` or ``briefcase dev``."""

from bubble_mark.ui.app import main

if __name__ == "__main__":
    main().main_loop()
