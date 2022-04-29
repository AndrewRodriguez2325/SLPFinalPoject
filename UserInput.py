import json
import pickle
import re
import string

import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import word_tokenize
from textblob import TextBlob


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class Meta(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return [{'pos': row['polarity'], 'sub': row['subjectivity'], 'len': row['len']} for _, row in data.iterrows()]


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
stop_words_list = set(stopwords.words('english'))
word_net_lemma = WordNetLemmatizer()


def used_text(token):
    return token not in stop_words_list and token not in list(string.punctuation) and len(token) > 2


def clean(article):
    article1 = []
    article2 = []
    # Replace(take out) "'" from article
    article = re.sub("'", "", article)
    # Replace(take out)  from article
    article = re.sub("(\\d|\\W)+", " ", article)
    article1 = [word_net_lemma.lemmatize(word, pos="v") for word in word_tokenize(article.lower()) if used_text(word)]
    article2 = [word for word in article1 if used_text(word)]
    return " ".join(article2)


def polarity(article):
    return TextBlob(article).sentiment[0]  # 0 negative - 1 positive


def subjectivity(article):
    return TextBlob(article).sentiment[1]  # 0 Fact - 1 Opinion


def length(article):
    if len(article.split()) > 0:
        return len(set(clean(article).split())) / len(article.split())
    else:
        return 0


def support_vector_classifier(text):
    clf_filename = 'support_vector_classifier.pkl'
    svm_clf = pickle.load(open(clf_filename, 'rb'))
    pipline_filename = 'pipeline.pkl'
    pipeline = pickle.load(open(pipline_filename, 'rb'))
    prediction = svm_clf.predict(text)
    print(prediction[0])


def lstm_classifier(text):
    clf_filename = 'lstm_classifier.pkl'
    lstm_clf = pickle.load(open(clf_filename, 'rb'))
    vectorizer_filename = 'vectorizer.pkl'
    vectorizer = pickle.load(open(vectorizer_filename, 'rb'))
    prediction = lstm_clf.predict(vectorizer.transform([text]))
    print(prediction[0])


if __name__ == "__main__":
    new_doc = "Chris Paul shot a perfect 14-of-14 from the field to help the Phoenix Suns close"
    polarity = polarity(new_doc)
    subject = subjectivity(new_doc)
    doc_len = length(new_doc)