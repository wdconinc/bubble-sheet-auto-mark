# Bubble Sheet Auto-Mark

A **privacy-first**, cross-platform application for grading bubble-sheet tests using a smartphone camera or desktop scanner.  All processing happens **on-device** – no images or grade data ever leave the device.

## Features

- **Live camera, recorded video, or imported images** – grade sheets in the field or at a desk
- **Automatic detection** – perspective correction and sheet normalisation via OpenCV
- **Student-ID recognition** – reads the bubble-coded ID grid from each sheet
- **Answer-key comparison** – configurable number of questions and choices (A–E)
- **CSV export** – `student_id,answers,score,num_correct,num_questions`
- **Cross-platform** – Linux, macOS, Windows, and Android (via Kivy + Buildozer)
- **MIT licensed**

## CSV Output Format

```csv
student_id,answers,score,num_correct,num_questions
123456789,13523422353322 1 35,0.857,12,14
987654321,M35 1422353322135,0.714,10,14
```

- Each character in `answers` is `1`–`5` (choice), ` ` (blank), or `M` (multiple marked).
- `score` is the fraction of *gradeable* (single-choice) answers that are correct.

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run on Desktop

```bash
python main.py
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Build for Android

```bash
buildozer android debug
```

## Project Layout

```
bubble-sheet-auto-mark/
├── main.py                      # App entry point
├── src/bubble_mark/
│   ├── processing/              # Image pipeline (OpenCV)
│   │   ├── image_utils.py       # Grayscale, threshold, perspective warp
│   │   ├── detector.py          # Sheet detection & bubble-grid layout
│   │   ├── analyzer.py          # Bubble fill analysis
│   │   └── grader.py            # Full grading pipeline
│   ├── models/
│   │   ├── answer_key.py        # AnswerKey (load/save/validate)
│   │   ├── grade_result.py      # GradeResult (score, CSV row)
│   │   └── settings.py          # AppSettings (layout config)
│   ├── export/
│   │   └── csv_exporter.py      # CSV export
│   └── ui/                      # Kivy screens (not imported by tests)
└── tests/                       # pytest test suite (106 tests)
```

## Settings

In the **Settings** screen you can configure:

| Setting | Default | Description |
|---|---|---|
| Number of questions | 30 | Questions per sheet |
| Number of choices | 5 | Choices per question (A–E) |
| ID digits | 9 | Digits in the student-ID field |
| Fill threshold | 0.5 | Dark-pixel fraction to call a bubble filled |
| Answer key | — | String of digits `1`–`5` or letters `A`–`E` |

## License

[MIT](LICENSE)
