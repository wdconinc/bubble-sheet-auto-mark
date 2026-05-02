---
layout: default
title: Installation
nav_order: 2
---

# Installation
{: .no_toc }

<details open markdown="block">
  <summary>Contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## Prerequisites

- **Python 3.12 or later**
- A working camera (built-in webcam, USB webcam, or smartphone) — or pre-captured images/video

### Python dependencies

| Package | Platform | Purpose |
|---|---|---|
| `toga>=0.4` | Desktop | Cross-platform UI framework (BeeWare) |
| `toga-android>=0.4` | Android | Toga backend for Android |
| `opencv-python-headless>=4.8` | Desktop only | Image processing pipeline (optional — NumPy/Pillow fallback used when absent) |
| `numpy>=1.26` | All | Array operations |
| `Pillow>=10.0` | All | Image I/O helpers |

{: .note }
OpenCV is **not** available in the Android Briefcase environment (no compatible wheel for Chaquopy). The app falls back to a NumPy/Pillow implementation automatically, so grading still works but perspective-correction performance may differ.

---

## Desktop (Linux / macOS / Windows)

```bash
# 1. Clone the repository
git clone https://github.com/wdconinc/bubble-sheet-auto-mark.git
cd bubble-sheet-auto-mark

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[toga,dev]"

# 4. Launch the app
briefcase dev
# or, after installation:
bubble-mark
```

---

## Android

Android builds use [Briefcase](https://briefcase.readthedocs.io/).

```bash
# Install Briefcase
pip install briefcase

# Build a debug APK
briefcase build android

# Deploy and run on a connected device
briefcase run android
```

All packaging configuration is in the `[tool.briefcase]` section of `pyproject.toml`.

{: .note }
Building for Android requires the Android SDK. Briefcase will prompt you to install it automatically on first build.

{: .warning }
OpenCV (`opencv-python-headless`) is **not** available in the Android build environment. The app automatically falls back to its NumPy/Pillow image-processing path, so grading still works, but some advanced detection features (e.g. contour overlay) are unavailable on Android.

---

## Development / Testing

Install the extra dev dependencies to run the test suite:

```bash
pip install -e ".[dev]"

# Run all tests
python -m pytest tests/ -v
```

All tests use synthetic images and do **not** require a camera or display. See the [project layout](configuration#project-layout) for details.
