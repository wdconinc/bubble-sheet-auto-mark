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

| Package | Purpose |
|---|---|
| `kivy==2.3.1` | Cross-platform UI framework |
| `opencv-python-headless>=4.8` | Image processing pipeline |
| `numpy>=1.26` | Array operations |
| `Pillow>=10.0` | Image I/O helpers |

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
pip install -r requirements.txt

# 4. Launch the app
python main.py
```

---

## Android

Android builds use [Buildozer](https://buildozer.readthedocs.io/).

```bash
# Install Buildozer (Linux / WSL recommended)
pip install buildozer

# Build a debug APK
buildozer android debug

# Deploy to a connected device
buildozer android debug deploy run
```

The `buildozer.spec` file in the repository root already contains the correct settings for the project.

{: .note }
Buildozer requires a Linux host (or WSL on Windows). The first build downloads the Android SDK and NDK, which takes some time.

---

## Development / Testing

Install the extra dev dependencies to run the test suite:

```bash
pip install -r requirements-dev.txt
# or
pip install -e ".[dev]"

# Run all tests
python -m pytest tests/ -v
```

All tests use synthetic images and do **not** require a camera or display. See the [project layout]({{ site.baseurl }}/#project-layout) for details.
