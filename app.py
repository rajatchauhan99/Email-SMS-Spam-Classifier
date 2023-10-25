import streamlit as st
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import nltk

ps = PorterStemmer()
nltk.download('stopwords')


def transform_text(text):
    text = text.lower()
    word_list = nltk.word_tokenize(text)
    new_list = []
    for i in word_list:
        if i.isalnum():
            new_list.append(i)
    new_list2 = []

    for i in new_list:
        if i not in stopwords.words("english") and i not in string.punctuation:
            new_list2.append(i)

    new_list3 = []

    for i in new_list2:
        i = ps.stem(i)
        new_list3.append(i)

    return " ".join(new_list3)

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the text: ")

if st.button("Predict"):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)

    # 2. vectorize
    vectorized_sms = tfidf.transform([transformed_sms])

    # 3. predict
    result = model.predict(vectorized_sms)[0]

    # 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
