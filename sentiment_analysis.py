import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import mlflow
import mlflow.sklearn

# NLTK setup
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv("data.csv")
df = df[df['Ratings'] != 3]
df['Sentiment'] = df['Ratings'].apply(lambda x: 1 if x >= 4 else 0)

# Text preprocessing
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['Clean_Review'] = df['Review text'].apply(clean_text)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['Clean_Review'])
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Flipkart Sentiment Analysis")

with mlflow.start_run(run_name="LogisticRegression_TFIDF_Run"):
 
    mlflow.set_tag("project", "Flipkart Sentiment Analysis")
    mlflow.set_tag("dataset", "Flipkart Product Reviews")
    mlflow.set_tag("algorithm", "Logistic Regression")
    mlflow.set_tag("feature_extraction", "TF-IDF")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_features", 5000)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "sentiment_model")

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    print("Training complete")
    print("F1 Score:", f1)
    print("Model and TF-IDF saved successfully")