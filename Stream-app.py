import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# ==== NLTK Resource Downloader ====
import nltk

def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except Exception as e:
        print(f"NLTK download error: {e}")

download_nltk_resources()


# ==== Streamlit UI ====
st.set_page_config(page_title="NLP Sentiment Analysis", layout="centered")
st.title("APPLICATION OF NLP FOR E-COMMERCE REVIEW SENTIMENT")

# ==== Preprocessing Tools ====
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def rewhitespace(text):
    corrected = str(text)
    corrected = re.sub(r"( )\1+", r"\1", corrected)
    corrected = re.sub(r"(\n)\1+", r"\1", corrected)
    corrected = re.sub(r"(\r)\1+", r"\1", corrected)
    corrected = re.sub(r"(\t)\1+", r"\1", corrected)
    return corrected.strip()

def token(text):
    return word_tokenize(text)

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_review(review):
    return [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in review]

# ==== Load Trained Vectorizer & Model ====
with open('tfidf_vectorizer.sav', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
st.sidebar.success("TF-IDF vectorizer loaded.")

with open('logistic_regression_model.sav', 'rb') as f:
    model = pickle.load(f)
st.sidebar.success("Logistic Regression model loaded.")

# ==== Input UI ====
st.subheader("Enter an E-Commerce Product Review")
content = st.text_area("Input your review here:")

# ==== Prediction ====
if st.button("Classify"):
    if content.strip():
        try:
            cleaned = clean_text(content)
            spaced = rewhitespace(cleaned)
            tokens = token(spaced)
            lemmatized = lemmatize_review(tokens)
            final_text = ' '.join(lemmatized)

            text_tfidf = tfidf_vectorizer.transform([final_text])
            prediction = model.predict(text_tfidf)[0]

            if prediction == 'positif':
                st.success('The sentiment is POSITIVE.')
            elif prediction == 'netral':
                st.info('The sentiment is NEUTRAL.')
            elif prediction == 'negatif':
                st.warning('The sentiment is NEGATIVE.')
            else:
                st.error('Unexpected prediction result.')

        except Exception as e:
            st.error(f"An error occurred during classification: {str(e)}")
    else:
        st.warning("Please enter a review to classify.")

# ==== Footer ====
st.markdown("---")
st.markdown("© 2025 NLP Sentiment Analysis. All rights reserved.")
