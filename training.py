import pandas as pd
import numpy as np
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# ---------- CLEANING FUNCTION ----------
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText


# ---------- LOAD TRAINING DATA ----------
df = pd.read_csv("/Users/pravinkumarrai/Desktop/UpdatedResumeDataSet.csv")
df["cleaned"] = df["Resume"].apply(cleanResume)

# Encode labels
le = LabelEncoder()
df["category"] = le.fit_transform(df["Category"])

# TF-IDF Vectorization
tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english")
X = tfidf.fit_transform(df["cleaned"])
y = df["category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Accuracy
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# Save files
pickle.dump(tfidf, open("tfidf.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))

print("Training Complete. Files saved: model.pkl, tfidf.pkl, encoder.pkl")
