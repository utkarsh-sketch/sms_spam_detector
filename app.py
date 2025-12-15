import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ------------------ DOWNLOAD NLTK DATA (SAFE FOR STREAMLIT CLOUD) ------------------
@st.cache_resource
def download_nltk_data():
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")

download_nltk_data()
# -------------------------------------------------------------------------------

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english"):
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# ------------------ LOAD MODEL & VECTORIZER ------------------
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
# ------------------------------------------------------------

st.title("📩 Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message")
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. display
        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")