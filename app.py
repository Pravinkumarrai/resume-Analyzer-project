import streamlit as st
import pickle
import docx
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pymongo import MongoClient
import pdfplumber

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    layout="wide"
)

# ---------------- MONGODB ----------------
client = MongoClient("mongodb://localhost:27017/")
db = client["resume_db"]
collection = db["resumes"]

# ---------------- LOAD MODEL ----------------
svc_model = pickle.load(open("model.pkl","rb"))
tfidf = pickle.load(open("tfidf.pkl","rb"))
le = pickle.load(open("encoder.pkl","rb"))

# ---------------- CLEAN TEXT ----------------
def cleanResume(txt):

    txt = txt.lower()

    txt = re.sub(r'http\S+', ' ', txt)

    txt = re.sub(r'[^a-zA-Z ]', ' ', txt)

    txt = re.sub(r'\s+', ' ', txt)

    return txt


# ---------------- EXTRACT TEXT ----------------
def extract_text_from_pdf(file):

    text=""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:

            text+=page.extract_text() or ""

    return text


def extract_text_from_docx(file):

    doc=docx.Document(file)

    return "\n".join([p.text for p in doc.paragraphs])


def extract_text_from_txt(file):

    return file.read().decode("utf-8",errors="ignore")


# ---------------- EXTRACT INFO ----------------
def extract_email(text):

    match=re.findall(r'\S+@\S+',text)

    return match[0] if match else "Not found"


def extract_phone(text):

    match=re.findall(r'\b\d{10}\b',text)

    return match[0] if match else "Not found"


def extract_name(text):

    blacklist=[
        "resume","curriculum","vitae",
        "profile","contact","email",
        "phone","education","skills",
        "experience","objective"
    ]

    lines=text.split("\n")

    for line in lines[:10]:

        line=line.strip()

        if len(line)==0:
            continue

        if any(word in line.lower() for word in blacklist):
            continue

        if any(char.isdigit() for char in line):
            continue

        if "@" in line:
            continue

        words=line.split()

        if 2<=len(words)<=4:

            if all(w.isalpha() for w in words):

                return line.title()

    return "Unknown"


# ---------------- CATEGORY ----------------
def predict_category(text):

    text_lower=text.lower()

    if "python" in text_lower or "machine learning" in text_lower:
        return "Data Science"

    if "react" in text_lower or "javascript" in text_lower:
        return "Web Developer"

    if "java" in text_lower:
        return "Java Developer"

    if "recruitment" in text_lower or "talent acquisition" in text_lower:
        return "HR"

    cleaned=cleanResume(text)

    vec=tfidf.transform([cleaned])

    pred=svc_model.predict(vec)

    return le.inverse_transform(pred)[0]


# ---------------- SKILLS ----------------
skill_keywords={

"Programming":["python","java","c++","sql"],

"ML":["machine learning","tensorflow","pytorch"],

"Web":["react","node","django","flask"],

"Cloud":["aws","docker","kubernetes"]

}

def analyze_skills(text):

    counts={k:0 for k in skill_keywords}

    text=text.lower()

    for cat,words in skill_keywords.items():

        for w in words:

            if w in text:

                counts[cat]+=1

    return pd.DataFrame({

        "Category":list(counts.keys()),

        "Count":list(counts.values())

    })


# ---------------- ACHIEVEMENTS ----------------
achievement_keywords=[

"award","certification",
"winner","rank","hackathon"

]

def analyze_achievements(text):

    text=text.lower()

    count=sum(text.count(w) for w in achievement_keywords)

    total=len(text.split())

    other=max(total/100-count,1)

    df=pd.DataFrame({

        "Type":["Achievements","Other"],

        "Count":[count,other]

    })

    return df,count


# ---------------- SCORE ----------------
def resume_score(text,skill_df,ach_df):

    score=0

    suggestions=[]

    if len(text.split())>400:
        score+=20
    else:
        suggestions.append("Add more content")

    if skill_df["Count"].sum()>=4:
        score+=30
    else:
        suggestions.append("Add more skills")

    if ach_df["Count"].iloc[0]>0:
        score+=20
    else:
        suggestions.append("Add achievements")

    if "project" in text.lower():
        score+=20
    else:
        suggestions.append("Add projects")

    if "education" in text.lower():
        score+=10
    else:
        suggestions.append("Add education")

    return score,suggestions


# ---------------- SCORE UI ----------------
def display_score(score):

    if score>=80:
        color="green"
        label="Excellent"
    elif score>=60:
        color="orange"
        label="Good"
    else:
        color="red"
        label="Needs Improvement"

    st.markdown(

        f"""

        <div style="
        background-color:#f5f7fa;
        padding:25px;
        border-radius:15px;
        text-align:center">

        <h1 style="color:{color};font-size:55px">
        {score}/100
        </h1>

        <h3 style="color:gray">
        {label}
        </h3>

        </div>

        """,

        unsafe_allow_html=True

    )

    st.progress(score/100)


# ---------------- APP ----------------
def main():

    st.set_page_config(
        page_title="AI Resume Analyzer",
        layout="wide"
    )

    st.title("AI Resume Analyzer")

    uploaded=st.file_uploader(

        "Upload Resume",

        type=["pdf","docx","txt"]

    )

    if uploaded:

        ext=uploaded.name.split(".")[-1]

        if ext=="pdf":
            text=extract_text_from_pdf(uploaded)

        elif ext=="docx":
            text=extract_text_from_docx(uploaded)

        else:
            text=extract_text_from_txt(uploaded)

        st.success("Resume Uploaded")

        col1,col2=st.columns(2)

        name=extract_name(text)

        if name=="Unknown":
            name=uploaded.name.replace(".pdf","")

        email=extract_email(text)

        phone=extract_phone(text)

        with col1:

            st.write("Name:",name)

            st.write("Email:",email)

            st.write("Phone:",phone)

        category=predict_category(text)

        with col2:

            st.info("Category: "+category)

        skill_df=analyze_skills(text)

        st.subheader("Skill Analysis")

        st.bar_chart(

            skill_df.set_index("Category")

        )

        ach_df,ach_count=analyze_achievements(text)

        score,suggestions=resume_score(

            text,

            skill_df,

            ach_df

        )

        st.subheader("Resume Score")

        display_score(score)

        st.subheader("Suggestions")

        for s in suggestions:

            st.write("-",s)

        existing=collection.find_one({

            "file_name":uploaded.name

        })

        if not existing:

            data={

                "name":name,

                "email":email,

                "phone":phone,

                "file_name":uploaded.name,

                "category":category,

                "skills":skill_df.to_dict(

                    orient="records"

                ),

                "score":score,

                "date":datetime.utcnow()

            }

            collection.insert_one(data)

            st.success("Saved to MongoDB")

    # dashboard
    st.divider()

    st.subheader("Stored Resumes")

    records=list(

        collection.find({},{"_id":0})

    )

    if records:

        df=pd.DataFrame(records)

        if "skills" in df.columns:

            df["skills"]=df["skills"].apply(

                lambda x:", ".join(

                    [

                        f"{i['Category']}({i['Count']})"

                        for i in x

                    ]

                )

            )

        st.dataframe(df)

        # leaderboard
        st.subheader("Top Candidates")

        top=df.sort_values(

            "score",

            ascending=False

        ).head(5)

        st.table(

            top[["name","category","score"]]

        )

        # download
        csv=df.to_csv(index=False)

        st.download_button(

            "Download Report",

            csv,

            "resume_data.csv"

        )

    else:

        st.info("No data available")


if __name__=="__main__":

    main()
