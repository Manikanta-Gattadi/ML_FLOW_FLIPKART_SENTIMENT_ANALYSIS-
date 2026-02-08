import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# NLTK setup
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Text preprocessing
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="Flipkart Sentiment Analysis", layout="centered")

st.title("Flipkart Review Sentiment Analyzer")
st.write("Enter a product review to predict its sentiment.")

review = st.text_area("Enter Review Text")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned_review = clean_text(review)
        vector = tfidf.transform([cleaned_review])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("Positive Review.")
        else:
            st.error("Negative Review.")