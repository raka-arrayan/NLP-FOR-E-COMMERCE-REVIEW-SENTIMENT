# E-Commerce Review Sentiment Analyzer

A simple web app built with **Streamlit** that uses **Natural Language Processing (NLP)** to analyze and classify the sentiment of e-commerce product reviews into **Positive**, **Neutral**, or **Negative**.

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live--App-brightgreen?logo=streamlit)](https://e-commerce-review-sentiment.streamlit.app/)


## Features

- Preprocessing: Text cleaning, lowercase conversion, removal of punctuation, numbers, and stopwords, whitespace cleanup, tokenization, and lemmatization.
- Model Training: Uses a pre-trained **TF-IDF Vectorizer** and a **Logistic Regression** classifier.
- UI: Clean and interactive interface using Streamlit.
- Live sentiment prediction for user-inputted product reviews.

---

## NLP Workflow

1. **Text Cleaning**
   - Lowercase conversion
   - Punctuation and symbol removal
   - Stopword removal

2. **Text Normalization**
   - Extra whitespace, tab, and newline cleanup

3. **Tokenization & Lemmatization**
   - Word tokenization
   - POS-tagged lemmatization using WordNet

4. **Vectorization**
   - Using saved `tfidf_vectorizer.sav`

5. **Sentiment Classification**
   - Using trained `logistic_regression_model.sav`

---

## How to Run

1. Clone the repository.
2. Make sure Python and the required packages are installed.
3. Run the Streamlit app
   ```
   streamlit run Stream-app.py
   ```
4. Ensure the following files are present in the same directory
   ```
    tfidf_vectorizer.sav
    logistic_regression_model.sav
   ```

---
**Raka Arrayan Muttaqien**  
© 2025 E-Commerce Sentiment Analyzer. All rights reserved.
