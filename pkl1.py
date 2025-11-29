try:
    import pandas as pd
    import numpy as np
    import re
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import pickle
except Exception as e:
    raise ImportError(
        "Missing required packages (e.g. pandas, numpy, scikit-learn). "
        "Install them with: python3 -m pip install pandas numpy scikit-learn\n"
        "Original import error: {}".format(e)
    )

# ----------------------------------------
# 1. Clean Resume Function
# ----------------------------------------
def cleanresume(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\b(?:https?://|www\.)[^\s]+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

# ----------------------------------------
# 2. Load Dataset
# ----------------------------------------
df = pd.read_csv("UpdatedResumeDataSet.csv")

df["Cleaned_Resume"] = df["Resume"].apply(cleanresume)

# ----------------------------------------
# 3. Encode Labels
# ----------------------------------------
le = LabelEncoder()
df["Category_Encoded"] = le.fit_transform(df["Category"])

# ----------------------------------------
# 4. TF-IDF Vectorizer
# ----------------------------------------
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df["Cleaned_Resume"])
y = df["Category_Encoded"]

# ----------------------------------------
# 5. Split dataset
# ----------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------
# 6. Train Model (KNN)
# ----------------------------------------
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(x_train, y_train)

# ----------------------------------------
# 7. Evaluate
# ----------------------------------------
ypred = clf.predict(x_test)
print("Accuracy:", accuracy_score(y_test, ypred) * 100)

# ----------------------------------------
# 8. Save Pickle Files
# ----------------------------------------
pickle.dump(clf, open("clf.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))

print("✔ All pickle files saved successfully!")
