# HACKATHON_FULL_UPGRADED.py
import streamlit as st
import docx2txt
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import uuid
import sqlite3
import os
from datetime import datetime
import altair as alt

# -----------------------------
# Config / OpenAI toggle
# -----------------------------
USE_OPENAI = False
try:
    if USE_OPENAI:
        from openai import OpenAI
        OPENAI_KEY = os.getenv("OPENAI_API_KEY")
        if OPENAI_KEY:
            client = OpenAI(api_key=OPENAI_KEY)
        else:
            st.warning("OPENAI_API_KEY not set. Falling back to local similarity.")
            USE_OPENAI = False
except Exception:
    USE_OPENAI = False

# -----------------------------
# Setup SQLite & Uploads
# -----------------------------
UPLOAD_DIR = "uploads"
DB_PATH = "resume_data.db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

def init_db():
    c.execute('''CREATE TABLE IF NOT EXISTS resumes
                 (id TEXT PRIMARY KEY, filename TEXT, filepath TEXT, uploaded_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS jds
                 (id TEXT PRIMARY KEY, filename TEXT, filepath TEXT, uploaded_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS evaluations
                 (id TEXT PRIMARY KEY,
                  resume_id TEXT,
                  jd_id TEXT,
                  score REAL,
                  verdict TEXT,
                  missing_skills TEXT,
                  feedback TEXT,
                  evaluated_at TEXT)''')
    conn.commit()
init_db()

# -----------------------------
# Helper Functions
# -----------------------------
def extract_text_from_pdf(pdf_file_obj):
    text = ""
    doc = fitz.open(stream=pdf_file_obj.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(docx_path_or_obj):
    try:
        return docx2txt.process(docx_path_or_obj)
    except Exception:
        with open(docx_path_or_obj, "rb") as f:
            return docx2txt.process(f)

def clean_text(text):
    text = (text or "").lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text

COMMON_SKILLS = [
    "python", "sql", "pandas", "numpy", "power bi", "tableau", "excel",
    "spark", "pyspark", "kafka", "c++", "scikit-learn", "tensorflow",
    "pytorch", "nlp", "computer vision", "generative ai", "machine learning",
    "deep learning", "statistics", "data analysis", "data visualization"
]

must_have_skills = ["python", "sql", "pandas", "numpy"]
nice_to_have_skills = ["spark", "pyspark", "kafka", "tableau", "power bi", "excel"]

def extract_keywords(text):
    text = clean_text(text)
    return [skill for skill in COMMON_SKILLS if skill in text]

def hard_match_weighted(resume_keywords):
    matches_must = [kw for kw in resume_keywords if kw in must_have_skills]
    matches_nice = [kw for kw in resume_keywords if kw in nice_to_have_skills]
    hard_score = 0.0
    if must_have_skills:
        hard_score += 0.7 * (len(matches_must) / len(must_have_skills) * 100)
    if nice_to_have_skills:
        hard_score += 0.3 * (len(matches_nice) / len(nice_to_have_skills) * 100)
    return round(hard_score, 2), matches_must + matches_nice

def soft_match_embeddings(resume_text, jd_text):
    if USE_OPENAI:
        res_emb = client.embeddings.create(model="text-embedding-3-large", input=resume_text)['data'][0]['embedding']
        jd_emb = client.embeddings.create(model="text-embedding-3-large", input=jd_text)['data'][0]['embedding']
        sim = float(np.dot(res_emb, jd_emb) / (np.linalg.norm(res_emb) * np.linalg.norm(jd_emb))) * 100
        return round(sim, 2)
    else:
        vect = CountVectorizer().fit([resume_text, jd_text])
        vecs = vect.transform([resume_text, jd_text])
        return round(cosine_similarity(vecs[0], vecs[1])[0][0] * 100, 2)

def get_verdict(score):
    if score >= 75:
        return "High"
    elif score >= 50:
        return "Medium"
    else:
        return "Low"

def generate_feedback(resume_text, missing_skills):
    if not missing_skills:
        return "Strong alignment! No major skill gaps."
    if USE_OPENAI:
        prompt = f"You are a career advisor.\nResume excerpt: {resume_text[:1000]}\nMissing Skills: {', '.join(missing_skills)}\nProvide short actionable feedback."
        try:
            resp = client.chat.completions.create(model="gpt-4", messages=[{"role":"user","content":prompt}], temperature=0.3)
            return resp.choices[0].message.content if hasattr(resp, "choices") else str(resp)
        except Exception:
            return f"Consider adding or improving skills: {', '.join(missing_skills)}"
    else:
        return f"Consider adding or improving skills: {', '.join(missing_skills)}"

def evaluate_resume(resume_text, jd_text):
    resume_keywords = extract_keywords(resume_text)
    hard_score, matched_skills = hard_match_weighted(resume_keywords)
    soft_score = soft_match_embeddings(clean_text(resume_text), clean_text(jd_text))
    final_score = round(0.6 * hard_score + 0.4 * soft_score, 2)
    verdict = get_verdict(final_score)
    jd_keywords = extract_keywords(jd_text)
    missing_skills = [kw for kw in jd_keywords if kw not in resume_keywords]
    feedback = generate_feedback(resume_text, missing_skills)
    return {
        "Relevance Score": final_score,
        "Verdict": verdict,
        "Resume Keywords": resume_keywords,
        "JD Keywords": jd_keywords,
        "Missing Skills": missing_skills,
        "Feedback": feedback
    }

def save_evaluation_to_db(resume_id, jd_id, score, verdict, missing_skills, feedback):
    evaluated_at = datetime.utcnow().isoformat()
    eval_id = str(uuid.uuid4())
    c.execute("INSERT INTO evaluations (id, resume_id, jd_id, score, verdict, missing_skills, feedback, evaluated_at) VALUES (?,?,?,?,?,?,?,?)",
              (eval_id, resume_id, jd_id, score, verdict, ", ".join(missing_skills), feedback, evaluated_at))
    conn.commit()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Dashboard / DB")
st.sidebar.markdown("💡 Tip: Upload JDs and Resumes in the main area, click Evaluate. The DB is auto-created & visible here.")

# Always show DB
try:
    df_db = pd.read_sql_query(
        "SELECT e.id AS eval_id, r.filename AS resume_file, j.filename AS jd_file, e.score, e.verdict, e.missing_skills, e.feedback, e.evaluated_at FROM evaluations e LEFT JOIN resumes r ON e.resume_id=r.id LEFT JOIN jds j ON e.jd_id=j.id ORDER BY e.evaluated_at DESC LIMIT 200", conn)
    st.sidebar.dataframe(df_db)
    csv_db = df_db.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button("Download DB CSV", csv_db, "evaluations_db.csv", "text/csv")
except:
    st.sidebar.write("No evaluations yet.")

# -----------------------------
# Main UI
# -----------------------------
st.markdown("""
<style>
/* Morph title animation */
@keyframes drawTitle {
  0% {clip-path: circle(0% at 50% 50%);}
  100% {clip-path: circle(150% at 50% 50%);}
}
.morph-title {
  font-size: 48px;
  font-weight: 800;
  color:white;
  animation: drawTitle 1.5s ease-out forwards;
}
.jd-box {
  background:#333; color:white; padding:18px; border-radius:12px; transition: transform 0.3s;
}
.jd-box:hover {
  transform: scale(1.03);
}
.animated-dot {
  width:8px;height:8px;border-radius:50%;background:#16a34a;position:absolute;animation:floatDot 5s infinite;
}
@keyframes floatDot {
  0%{transform: translate(0,0);}
  50%{transform: translate(200px,100px);}
  100%{transform: translate(0,0);}
}
</style>
<h1 class='morph-title'>🚀 Automated Resume Relevance Checker</h1>
<p style="color:#ccc;">Made by Anant Sharma</p>
<p style="color:#aaa;">⚠️ Please view in dark mode for better results.</p>
""", unsafe_allow_html=True)

# Random animated dots
for i in range(5):
    st.markdown(f"<div class='animated-dot' style='top:{i*50}px;left:{i*100}px;'></div>", unsafe_allow_html=True)

# -----------------------------
# JD Upload
# -----------------------------
jd_files = st.file_uploader("📄 Upload Job Description(s) (PDF/DOCX/TXT) or paste text", type=["pdf","docx","txt"], accept_multiple_files=True)
jd_texts = []
jd_ids = []

for jd_file in jd_files:
    jd_id = str(uuid.uuid4())
    filename = jd_file.name
    dest_path = os.path.join(UPLOAD_DIR, f"{jd_id}_{filename}")
    with open(dest_path, "wb") as f:
        f.write(jd_file.read())
    uploaded_at = datetime.utcnow().isoformat()
    c.execute("INSERT INTO jds (id, filename, filepath, uploaded_at) VALUES (?, ?, ?, ?)", (jd_id, filename, dest_path, uploaded_at))
    conn.commit()
    jd_ids.append(jd_id)
    if jd_file.type == "application/pdf":
        jd_texts.append(extract_text_from_pdf(open(dest_path, "rb")))
    elif jd_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        jd_texts.append(extract_text_from_docx(dest_path))
    elif jd_file.type == "text/plain":
        with open(dest_path, "r", encoding="utf-8") as f:
            jd_texts.append(f.read())

if st.checkbox("Paste JD manually"):
    pasted = st.text_area("Paste JD text here")
    if pasted.strip():
        jd_texts.append(pasted)
        jd_ids.append(str(uuid.uuid4()))

st.markdown("---")

# -----------------------------
# Resume Upload
# -----------------------------
resume_files = st.file_uploader("📎 Upload Resume(s) (PDF/DOCX)", type=["pdf","docx"], accept_multiple_files=True)
resume_upload_records = []
st.info("Tip: Upload multiple resumes and JDs. CSV export available after evaluation.")

# Verdict filter
verdict_filter = st.multiselect("Filter by Verdict", options=["High", "Medium", "Low"], default=["High", "Medium", "Low"])

# JD filter
jd_filter = st.multiselect("Filter by JD", options=[f"JD {i+1}" for i in range(len(jd_texts))], default=[f"JD {i+1}" for i in range(len(jd_texts))])

# -----------------------------
# Evaluate Button
# -----------------------------
if st.button("✅ Evaluate Resumes") and jd_texts and resume_files:
    results = []
    for file in resume_files:
        resume_id = str(uuid.uuid4())
        filename = file.name
        dest_path = os.path.join(UPLOAD_DIR, f"{resume_id}_{filename}")
        with open(dest_path, "wb") as f:
            f.write(file.read())
        uploaded_at = datetime.utcnow().isoformat()
        c.execute("INSERT INTO resumes (id, filename, filepath, uploaded_at) VALUES (?, ?, ?, ?)", (resume_id, filename, dest_path, uploaded_at))
        conn.commit()
        resume_upload_records.append({"id": resume_id, "filename": filename, "filepath": dest_path})
        if file.type == "application/pdf":
            resume_text = extract_text_from_pdf(open(dest_path, "rb"))
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_docx(dest_path)
        else:
            resume_text = ""

        jd_results = []
        for i, jd_text in enumerate(jd_texts):
            eval_res = evaluate_resume(resume_text, jd_text)
            eval_res["JD"] = f"JD {i+1}"
            eval_res["Resume ID"] = resume_id
            eval_res["JD ID"] = jd_ids[i] if i < len(jd_ids) else str(uuid.uuid4())
            jd_results.append(eval_res)
            save_evaluation_to_db(eval_res["Resume ID"], eval_res["JD ID"], eval_res["Relevance Score"], eval_res["Verdict"], eval_res["Missing Skills"], eval_res["Feedback"])
        results.append({"Candidate": filename, "JD Results": jd_results, "Resume File": dest_path})

    # Prepare DataFrame for chart
    df_rows = []
    for r in results:
        avg_score = np.mean([jd_r["Relevance Score"] for jd_r in r["JD Results"]])
        for jd_r in r["JD Results"]:
            if jd_r["Verdict"] in verdict_filter and jd_r["JD"] in jd_filter:
                df_rows.append({
                    "Candidate": r["Candidate"],
                    "Job": jd_r["JD"],
                    "Score": jd_r["Relevance Score"],
                    "Verdict": jd_r["Verdict"],
                    "Average": avg_score
                })
    df_results = pd.DataFrame(df_rows)
    df_results = df_results.sort_values(by="Average", ascending=False)

    # Summary Stats
    st.markdown("### Summary Stats")
    st.write(f"**Total Resumes Evaluated:** {len(resume_files)}")
    for v in ["High","Medium","Low"]:
        count = df_results[df_results["Verdict"]==v].shape[0]
        st.markdown(f"[{v} Fit: {count}](#)")

    # Charts
    tab1, tab2 = st.tabs(["📈 Summary", "📝 Detailed Feedback"])

    with tab1:
        for jd_index in range(len(jd_texts)):
            df_chart = df_results[df_results["Job"]==f"JD {jd_index+1}"].copy()
            if not df_chart.empty:
                df_chart["Candidate_JD"] = df_chart["Candidate"] + " | " + df_chart["Job"]
                color_scale = alt.Scale(domain=["High","Medium","Low"], range=["#16a34a","#f59e0b","#ef4444"])
                chart = alt.Chart(df_chart).mark_bar().encode(
                    x=alt.X("Candidate_JD", sort=None, title="Candidate | Job"),
                    y=alt.Y("Score", title="Relevance Score"),
                    color=alt.Color("Verdict", scale=color_scale),
                    tooltip=["Candidate","Job","Score","Verdict"]
                ).properties(width=1000, height=450, title=f"Resume Relevance Scores - JD {jd_index+1}")
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info(f"No results for JD {jd_index+1} (check filters)")

    with tab2:
        for r in results:
            with st.expander(f"{r['Candidate']}", expanded=True):
                avg_score = np.mean([jd_r["Relevance Score"] for jd_r in r["JD Results"]])
                st.markdown(f"<h3 style='text-align:center;'>Average Score: {avg_score:.2f}</h3>", unsafe_allow_html=True)
                cols = st.columns(len(r["JD Results"]))
                for i, jd_r in enumerate(r["JD Results"]):
                    with cols[i]:
                        st.markdown(f"<div class='jd-box'><strong>{jd_r['JD']}</strong>", unsafe_allow_html=True)
                        st.write(f"**Score:** {jd_r['Relevance Score']}")
                        st.write(f"**Verdict:** {jd_r['Verdict']}")
                        st.write(f"**Missing Skills:** {', '.join(jd_r['Missing Skills']) if jd_r['Missing Skills'] else 'None'}")
                        st.write(f"**Feedback:** {jd_r['Feedback']}")
                        st.markdown("</div>", unsafe_allow_html=True)

    if not df_results.empty:
        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results CSV", csv, "resume_evaluation_results.csv", "text/csv")

else:
    st.info("Upload at least one JD and one Resume, then click 'Evaluate Resumes'")
