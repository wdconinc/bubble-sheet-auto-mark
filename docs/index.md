---
layout: home
title: Home
nav_order: 1
---

# Bubble Sheet Auto-Mark

A **privacy-first**, cross-platform application for grading bubble-sheet tests using a smartphone camera or desktop scanner.

{: .highlight }
> **All processing happens on-device.** No images or grade data ever leave your device.

---

## Features

| Feature | Details |
|---|---|
| **Input sources** | Live camera, recorded video, or imported image files |
| **Automatic detection** | Perspective correction and sheet normalisation via OpenCV |
| **Student-ID recognition** | Reads the bubble-coded ID grid from each sheet |
| **Answer-key comparison** | Configurable number of questions and choices (A–E) |
| **CSV export** | One row per sheet: `student_id, answers, score, num_correct, num_questions` |
| **Cross-platform** | Linux, macOS, Windows, and Android (via Kivy + Buildozer) |
| **MIT licensed** | Free to use, modify, and distribute |

---

## Project Goals

Bubble Sheet Auto-Mark was designed for educators who need a **fast, reliable, and private** way to grade multiple-choice assessments without sending student data to third-party services.

Key design principles:

- **Privacy first** – all image processing and grading logic runs locally on the device.
- **No special hardware** – any modern smartphone or laptop camera is sufficient.
- **Configurable** – the number of questions, choices per question, and ID digits are all adjustable.
- **Testable** – the processing pipeline is pure Python (no UI dependency), enabling automated testing with synthetic images.

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

See the [Installation](installation) page for full prerequisites and the [Usage](usage) page to learn how to grade your first sheet.

---

## CSV Output

Each graded sheet produces one CSV row:

```csv
student_id,answers,score,num_correct,num_questions
123456789,13523422353322 1 35,0.857,12,14
987654321,M35 1422353322135,0.714,10,14
```

- Each character in `answers` is `1`–`5` (choice index), ` ` (blank/unanswered), or `M` (multiple bubbles marked).
- `score` is the fraction of *gradeable* (single-choice) answers that are correct.
- `score`, `num_correct`, and `num_questions` are omitted when no answer key has been set.
