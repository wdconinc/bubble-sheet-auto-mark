---
layout: default
title: Usage
nav_order: 3
---

# Usage
{: .no_toc }

<details open markdown="block">
  <summary>Contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## Launching the Application

```bash
python main.py
```

The app starts with a home screen. From here you can:

1. Configure settings (number of questions, answer key, etc.)
2. Start a grading session from the camera or an image file
3. Export results to CSV

---

## Grading Workflow

### 1. Configure Settings

Before grading, open the **Settings** screen to set the number of questions, number of choices, and (optionally) the answer key. See [Configuration](configuration) for all options.

### 2. Start a Grading Session

Choose your input source:

| Source | When to use |
|---|---|
| **Live camera** | Grade sheets in real time as you scan them |
| **Video file** | Process a pre-recorded video of sheets being scanned |
| **Image file(s)** | Import photos taken earlier |

### 3. Present Each Sheet

Hold or place each bubble sheet within the camera frame. The application will:

1. **Detect** the sheet boundary and apply perspective correction.
2. **Read** the student-ID bubble grid.
3. **Analyse** each answer bubble (filled vs. blank vs. multiple).
4. **Score** the sheet against the answer key (if one is configured).

A summary is displayed on screen for each detected sheet.

### 4. Export Results

When you are done, use the **Export** button to save all results to a CSV file. The file contains one row per graded sheet:

```
student_id,answers,score,num_correct,num_questions
```

---

## Understanding the Answers String

Each character in the `answers` column represents one question:

| Character | Meaning |
|---|---|
| `1` – `5` | The bubble that was filled (choice index) |
| ` ` (space) | No bubble was filled (blank / unanswered) |
| `M` | Multiple bubbles were filled (ambiguous) |

---

## Tips

- Ensure good, even lighting to improve bubble detection accuracy.
- Keep the sheet as flat as possible when photographing.
- Blank and `M` answers are *not* counted toward the score denominator — only unambiguously-filled bubbles are considered gradeable.
- You can re-export CSV at any time during a session without stopping grading.
