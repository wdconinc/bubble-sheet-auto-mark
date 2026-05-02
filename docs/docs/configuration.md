---
layout: default
title: Configuration
nav_order: 4
---

# Configuration
{: .no_toc }

<details open markdown="block">
  <summary>Contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## Settings Reference

All settings are accessible from the **Settings** screen in the application.

| Setting | Default | Description |
|---|---|---|
| **Number of questions** | `30` | Number of answer questions per sheet |
| **Number of choices** | `5` | Choices per question (A through E when 5) |
| **ID digits** | `9` | Number of digits in the student-ID bubble grid |
| **Fill threshold** | `0.5` | Dark-pixel fraction required to consider a bubble filled |
| **Answer key** | *(empty)* | String of digits `1`–`5` or letters `A`–`E`; one character per question |

---

## Answer Key Format

The answer key is a plain string — one character per question — where each character is either a digit (`1`–`5`) or a letter (`A`–`E`).

Examples for a 5-question test:

| Format | Meaning |
|---|---|
| `12345` | Q1=A, Q2=B, Q3=C, Q4=D, Q5=E |
| `ABCDE` | Same as above, using letter notation |
| `31524` | Q1=C, Q2=A, Q3=E, Q4=B, Q5=D |

{: .note }
Letters are case-insensitive and converted to the corresponding digit internally (`A`→`1`, `B`→`2`, …, `E`→`5`).

If no answer key is provided, the application still reads and exports student IDs and raw answers but omits the `score`, `num_correct`, and `num_questions` columns from the CSV.

---

## Fill Threshold

The **fill threshold** controls how dark a bubble must be (relative to its area) to count as filled.

- **Lower value** (e.g. `0.3`) – marks lighter pencil strokes as filled; may cause false positives.
- **Higher value** (e.g. `0.7`) – requires heavier shading; may miss lightly-filled bubbles.

The default of `0.5` works well under normal lighting conditions with standard pencil marks.

---

## Project Layout

```
bubble-sheet-auto-mark/
├── main.py                        # App entry point
├── src/bubble_mark/
│   ├── processing/                # Image pipeline (OpenCV)
│   │   ├── image_utils.py         # Grayscale, threshold, perspective warp
│   │   ├── detector.py            # Sheet detection & bubble-grid layout
│   │   ├── analyzer.py            # Bubble fill analysis
│   │   └── grader.py              # Full grading pipeline
│   ├── models/
│   │   ├── answer_key.py          # AnswerKey (load/save/validate)
│   │   ├── grade_result.py        # GradeResult (score, CSV row)
│   │   └── settings.py            # AppSettings (layout config)
│   ├── export/
│   │   └── csv_exporter.py        # CSV export
│   └── ui/                        # Toga screens (no test dependency)
└── tests/                         # pytest suite (~159 tests, no camera needed)
```
