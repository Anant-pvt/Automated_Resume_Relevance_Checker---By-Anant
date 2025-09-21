# 🚀 Automated Resume Relevance Checker

This is a Streamlit-based tool to automatically evaluate resumes against job descriptions (JDs) and provide relevance scores, verdicts, and skill feedback.

---

## Features
- Upload multiple resumes (PDF/DOCX) and JDs (PDF/DOCX/TXT or paste text)
- Evaluates resumes based on must-have and nice-to-have skills
- Calculates relevance scores (hard & soft match)
- Provides detailed feedback per resume per JD
- Charts & summaries for visualization
- Export results & DB as CSV

---

## Requirements
- Python 3.9+
- Streamlit
- pandas
- numpy
- scikit-learn
- docx2txt
- PyMuPDF (`fitz`)
- altair
- sqlite3 (built-in)
- re (built-in)
- uuid (built-in)
- os, datetime (built-in)

Install dependencies via:
```bash
pip install -r requirements.txt
