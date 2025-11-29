import streamlit as st
import pickle
import docx
import PyPDF2
import re
import matplotlib.pyplot as plt
import pandas as pd

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
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
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

# ---------------- SKILL & ACHIEVEMENT ANALYSIS ----------------
skill_keywords = {
    "Programming": ["python", "java", "c++", "javascript", "c#", "sql", "html", "css"],
    "Machine Learning": ["machine learning", "deep learning", "tensorflow", "pytorch", "data analysis", "model"],
    "Web Development": ["react", "node", "express", "django", "flask"],
    "Cloud & DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "devops", "ci/cd"],
    "Tools": ["git", "jira", "tableau", "power bi", "excel","weight loss","cardio","nutriton"]
}

achievement_keywords = ["award", "certification", "certificate", "achieved", "winner", "rank", "hackathon"]

def analyze_skills(text):
    counts = {cat: 0 for cat in skill_keywords}
    text = text.lower()
    for cat, keys in skill_keywords.items():
        for k in keys:
            if k in text:
                counts[cat] += 1
    df = pd.DataFrame({"Category": counts.keys(), "Count": counts.values()})
    return df

def analyze_achievements(text):
    text = text.lower()
    count = sum(text.count(word) for word in achievement_keywords)
    df = pd.DataFrame({"Type": ["Achievements", "Other Content"], "Count": [count, max(len(text)/100 - count, 1)]})
    return df

# ---------------- STREAMLIT APP ----------------
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")
    st.title("📄 Resume Category Prediction App")
    st.write("Upload a resume (PDF, DOCX, TXT) and get job category prediction with skill & achievement analysis.")

    uploaded = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

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
            st.subheader("Predicted Category:")
            category = pred(text)
            st.success(f"✔ {category}")

            # ---------- Skills Bar Chart ----------
            st.subheader("📊 Skills Detected in Resume")
            skill_df = analyze_skills(text)
            st.bar_chart(skill_df.set_index("Category"))

            # ---------- Achievements Pie Chart ----------
            st.subheader("🥇 Achievements Breakdown")
            ach_df = analyze_achievements(text)
            fig, ax = plt.subplots()
            ax.pie(ach_df["Count"], labels=ach_df["Type"], autopct="%.1f%%", colors=["#4CAF50", "#2196F3"])
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
