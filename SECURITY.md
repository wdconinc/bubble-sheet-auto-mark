# Security Policy

The maintainers of **Bubble Sheet Auto-Mark** take security seriously. This document explains
the scope of security considerations for this project and how to report issues.

## Scope

Bubble Sheet Auto-Mark is a fully offline, local application with no server, backend, or
network communication. Its attack surface is therefore limited to the device on which it runs.
Security issues relevant to this project include:

- **Malicious input files** – a crafted image or video file that causes unintended behaviour
  (e.g., a crash, excessive resource consumption, or arbitrary code execution via a library
  vulnerability).
- **Data leakage on shared devices** – unintended writing of student images or grade data to
  locations accessible by other users of the same device.
- **Privilege escalation** – any path by which the app could acquire permissions beyond those
  the user granted (e.g., broader file-system or camera access than needed).

Issues that are out of scope include vulnerabilities in the user's operating system, third-party
libraries (report those upstream), or configurations unrelated to this application.

## Reporting a vulnerability

Please open an issue in the
[GitHub repository](https://github.com/wdconinc/bubble-sheet-auto-mark/issues) describing:

1. A clear description of the issue.
2. Steps to reproduce it (including any crafted input files, if applicable).
3. The potential impact you see.

The maintainers will respond on a best-effort basis and aim to acknowledge reports promptly.

## Dependency security

This project uses [pip-audit](https://pypi.org/project/pip-audit/) or equivalent tooling in CI
to monitor for known vulnerabilities in dependencies. If you discover a vulnerability in a
dependency (OpenCV, Kivy, NumPy, Pillow, etc.), please also report it to the upstream project.
