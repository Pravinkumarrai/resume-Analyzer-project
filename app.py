import streamlit as st
import pickle
import docx
import PyPDF2
import re
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
from datetime import datetime

# -------- DATABASE --------
conn = sqlite3.connect("resumes.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS resumes(
id INTEGER PRIMARY KEY AUTOINCREMENT,
file_name TEXT,
category TEXT,
upload_date TEXT
)
""")

conn.commit()

# ---------------- LOAD TRAINED MODEL ----------------
svc_model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
le = pickle.load(open("encoder.pkl", "rb"))

# ---------------- CLEANING FUNCTION ----------------
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText


# ---------------- FILE EXTRACTORS ----------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text_from_txt(file):
    data = file.read()
    try:
        return data.decode("utf-8")
    except:
        return data.decode("latin-1")


# ---------------- PREDICTION ----------------
def pred(input_resume):
    cleaned = cleanResume(input_resume)
    vector = tfidf.transform([cleaned])
    result = svc_model.predict(vector)
    return le.inverse_transform(result)[0]


# ---------------- SKILL KEYWORDS ----------------
skill_keywords = {
    "Programming": ["python", "java", "c++", "javascript", "c#", "sql", "html", "css"],
    "Machine Learning": ["machine learning", "deep learning", "tensorflow", "pytorch", "data analysis"],
    "Web Development": ["react", "node", "express", "django", "flask"],
    "Cloud & DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "devops"],
    "Tools": ["git", "jira", "tableau", "power bi", "excel"]
}

achievement_keywords = [
    "award", "certification", "certificate",
    "achieved", "winner", "rank", "hackathon"
]


# ---------------- SKILL ANALYSIS ----------------
def analyze_skills(text):

    counts = {cat: 0 for cat in skill_keywords}

    text = text.lower()

    for cat, keys in skill_keywords.items():
        for k in keys:
            if k in text:
                counts[cat] += 1

    df = pd.DataFrame({
        "Category": counts.keys(),
        "Count": counts.values()
    })

    return df


# ---------------- ACHIEVEMENT ANALYSIS ----------------
def analyze_achievements(text):

    text = text.lower()

    achievement_count = sum(text.count(word) for word in achievement_keywords)

    total_words = len(text.split())

    other_content = max(total_words/100 - achievement_count, 1)

    df = pd.DataFrame({
        "Type": ["Achievements", "Other Content"],
        "Count": [achievement_count, other_content]
    })

    return df, achievement_count


# ---------------- RESUME SCORING ----------------
def resume_score(text, skill_df, achievement_df):

    score = 0
    suggestions = []

    word_count = len(text.split())

    if word_count > 500:
        score += 20
    else:
        suggestions.append("Add more details about projects and experience.")

    skill_count = skill_df["Count"].sum()

    if skill_count >= 5:
        score += 30
    else:
        suggestions.append("Include more technical skills.")

    ach_count = achievement_df["Count"].iloc[0]

    if ach_count > 0:
        score += 20
    else:
        suggestions.append("Add certifications, awards, or hackathon achievements.")

    if "project" in text.lower():
        score += 20
    else:
        suggestions.append("Mention at least one strong project.")

    if "university" in text.lower() or "college" in text.lower():
        score += 10
    else:
        suggestions.append("Include education details clearly.")

    return score, suggestions


# ---------------- STREAMLIT APP ----------------
def main():

    st.set_page_config(
        page_title="Resume Analyzer",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 AI Resume Analyzer")

    st.write(
        "Upload your resume to get job category prediction, skill insights, achievements analysis and improvement suggestions."
    )

    uploaded = st.file_uploader(
        "Upload Resume",
        type=["pdf", "docx", "txt"]
    )

    if uploaded:

        try:

            ext = uploaded.name.split(".")[-1].lower()

            if ext == "pdf":
                text = extract_text_from_pdf(uploaded)

            elif ext == "docx":
                text = extract_text_from_docx(uploaded)

            else:
                text = extract_text_from_txt(uploaded)

            st.success("File Uploaded Successfully")

            if st.checkbox("Show Extracted Resume Text"):
                st.text_area("Resume Text", text, height=300)

            # ---------- Prediction ----------
            st.subheader("Predicted Job Category")

            category = pred(text)

            st.success(f"✔ {category}")

            # ---------- Save to Database ----------
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute(
                "INSERT INTO resumes (file_name, category, upload_date) VALUES (?, ?, ?)",
                (uploaded.name, category, date)
            )

            conn.commit()

            # ---------- Skill Analysis ----------
            st.subheader("📊 Skills Detected")

            skill_df = analyze_skills(text)

            st.bar_chart(skill_df.set_index("Category"))

            # ---------- Achievement Analysis ----------
            st.subheader("🏆 Achievements Analysis")

            ach_df, ach_count = analyze_achievements(text)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Achievements Found", ach_count)

                if ach_count == 0:
                    st.warning("No achievements detected. Consider adding certifications or awards.")

            with col2:

                fig, ax = plt.subplots(figsize=(3,3))

                ax.pie(
                    ach_df["Count"],
                    labels=ach_df["Type"],
                    autopct="%1.0f%%",
                    startangle=90,
                    wedgeprops=dict(width=0.4)
                )

                ax.set_title("Achievement Distribution")

                st.pyplot(fig)

            # ---------- Resume Score ----------
            st.subheader("📈 Resume Score")

            score, suggestions = resume_score(text, skill_df, ach_df)

            st.progress(score/100)

            st.success(f"Resume Score: {score}/100")

            # ---------- Suggestions ----------
            st.subheader("💡 Resume Improvement Suggestions")

            if suggestions:
                for s in suggestions:
                    st.write(f"• {s}")
            else:
                st.success("Your resume looks strong!")

        except Exception as e:
            st.error(f"Error: {str(e)}")

    # ---------- Admin Dashboard ----------
    st.divider()

    st.subheader("📂 Uploaded Resume Records")

    data = cursor.execute("SELECT * FROM resumes").fetchall()

    if data:

        df = pd.DataFrame(
            data,
            columns=["ID", "File Name", "Category", "Upload Date"]
        )

        st.dataframe(df)

    else:
        st.info("No resumes uploaded yet.")


if __name__ == "__main__":
    main()