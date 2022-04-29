# Import packages and libraries needed
import pickle
import re
import string

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from sklearn.model_selection import cross_val_score
from keras.layers import Dense, LSTM
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from textblob import TextBlob

# Visualize the data and data types
df_News_Cat = pd.read_json("News_Category_Dataset_v2.json", lines=True)
print(df_News_Cat.columns)
print("---------------------------------")
df_News_Cat.info()
print("---------------------------------")
print(df_News_Cat['category'].value_counts())

# Data Clean up
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
stop_words_list = set(stopwords.words('english'))
word_net_lemma = WordNetLemmatizer()


# Function used to determine which tokens to use for cleanup (not stop words, not punctuation, not short words)
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


# We don't need every column from the JSON object we are really only interested in the header and short description3
df_News_Cat['data'] = df_News_Cat['headline'] + " " + df_News_Cat['short_description']
print("Original -> " + df_News_Cat.data[52])
print("Cleaned and processed -> " + clean(df_News_Cat.data[52]))


# See what words are most important using word cloud
def freq_words(x, terms=30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()

    freq_dist = nltk.FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(freq_dist.keys()), 'count': list(freq_dist.values())})
    fig = plt.figure(figsize=(21, 16))
    d = words_df.nlargest(columns="count", n=terms)
    ax2 = sns.barplot(data=d, palette=sns.color_palette('BuGn_r'), x="count", y="word")
    ax2.set(ylabel='Word')
    plt.show()


# df_News_Cat['cleaned_data'] = df_News_Cat['data'].apply(lambda x: clean(x))


# print(freq_words(df_News_Cat['data'], 25))
# print(freq_words(df_News_Cat['cleaned_data'], 25))
# subset = df_News_Cat[df_News_Cat.category == "CRIME"]
# text = subset.cleaned_data.values
# print(freq_words(text, 25))


# Create news variables (meta data) to improve classifier
def polarity(article):
    return TextBlob(article).sentiment[0]  # 0 negative - 1 positive


def subjectivity(article):
    return TextBlob(article).sentiment[1]  # 0 Fact - 1 Opinion


def length(article):
    if len(article.split()) > 0:
        return len(set(clean(article).split())) / len(article.split())
    else:
        return 0


# Adding columns to the data frame
df_News_Cat['polarity'] = df_News_Cat['data'].apply(polarity)
df_News_Cat['subjectivity'] = df_News_Cat['data'].apply(subjectivity)
df_News_Cat['len'] = df_News_Cat['data'].apply(length)


# Make class for feature union to apply different transformers to the whole of the input data and combine results
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


# Custom Pipeline for feature extraction
pipeline = Pipeline([
    ('union', FeatureUnion(
        transformer_list=[
            # Pipeline for extracting features from article using TF-IDF
            ('data', Pipeline([
                ('selector', ItemSelector(key='data')),
                ('tfidf', TfidfVectorizer(min_df=3, max_df=0.2, max_features=None,
                                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                                          ngram_range=(1, 10), use_idf=1, smooth_idf=1, sublinear_tf=1,
                                          stop_words=None, preprocessor=clean)),
            ])),

            # Pipeline for extracting meta data
            ('meta', Pipeline([
                ('selector', ItemSelector(key=['polarity', 'subjectivity', 'len'])),
                ('meta', Meta()),  # returns a list of feature-value mappings to vectors
                ('vect', DictVectorizer()),  # make the feature matrix
            ])),

        ],

        # Weigh the two pipelines accordingly
        transformer_weights={
            'data': 1,
            'meta': 1.5,
        },
    ))
])

# Build the pipeline and split data
seed = 52
mlX = df_News_Cat[['data', 'polarity', 'subjectivity', 'len']]  # What we provide
mly = df_News_Cat['category']  # What should be matched
ml_encoder = LabelEncoder()
y = ml_encoder.fit_transform(mly)
ml_x_train, ml_x_test, ml_y_train, ml_y_test = train_test_split(mlX, y, test_size=0.2, random_state=seed, stratify=y)
# Fit all the transformers one after the other and transform the data.
# Finally, fit the transformed data using the final estimator.
pipeline.fit(ml_x_train)
train_vec = pipeline.transform(ml_x_train)
test_vec = pipeline.transform(ml_x_test)

support_vector = LinearSVC(C=1, class_weight='balanced', multi_class='ovr', random_state=52,
                           max_iter=10000)  # Support Vector machines
support_vector.fit(train_vec, ml_y_train)
SVM_prediction = support_vector.predict(test_vec)
# svm_clf_filename = 'support_vector_classifier.pkl'
# pickle.dump(support_vector, open(svm_clf_filename, 'wb'))
# pipeline_filename = 'pipeline.pkl'
# pickle.dump(pipeline, open(pipeline_filename, 'wb'))
# svm_scores = cross_val_score(support_vector, train_vec, y_train, cv=3, scoring="accuracy")
# print(svm_scores)
print('SVM Accuracy: ', accuracy_score(ml_y_test, SVM_prediction))
print(classification_report(ml_y_test, SVM_prediction))

stochastic_gradient = SGDClassifier(max_iter=200, )  # Stochastic Gradient Classifier
stochastic_gradient.fit(train_vec, ml_y_train)
SGC_prediction = stochastic_gradient.predict(test_vec)
# sgc_scores = cross_val_score(stochastic_gradient, train_vec, y_train, cv=2, scoring="accuracy")
# print(sgc_scores)
print('SGC Accuracy: ', accuracy_score(ml_y_test, SGC_prediction))
print(classification_report(ml_y_test, SGC_prediction))

# Attempt to implement a simple LSTM
slp = spacy.load('en_core_web_lg')  # Load spaCy embeddings
dlX = df_News_Cat['data']  # Just using the text here
dly = df_News_Cat['category']
encoder = LabelEncoder()
dly = encoder.fit_transform(dly)
Y = np_utils.to_categorical(dly)
# Create the tf-idf vector
vectorizer = TfidfVectorizer(min_df=3, max_df=0.2, max_features=None,
                             strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                             use_idf=1, smooth_idf=1, sublinear_tf=1,
                             stop_words=None, preprocessor=clean)
seed = 52
x_train, x_test, y_train, y_test = train_test_split(dlX, Y, test_size=0.2, random_state=seed, stratify=y)
vectorizer.fit(x_train)
word2idx = {word: idx for idx, word in enumerate(vectorizer.get_feature_names())}
tokenize = vectorizer.build_tokenizer()
preprocess = vectorizer.build_preprocessor()


def to_sequence(tokenizer, preprocessor, index, text):
    words = tokenizer(preprocessor(text))
    indexes = [index[word] for word in words if word in index]
    return indexes


X_train_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in x_train]
MAX_SEQ_LENGTH = 60

N_FEATURES = len(vectorizer.get_feature_names())
X_train_sequences = pad_sequences(X_train_sequences, maxlen=MAX_SEQ_LENGTH, value=N_FEATURES)
X_test_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in x_test]
X_test_sequences = pad_sequences(X_test_sequences, maxlen=MAX_SEQ_LENGTH, value=N_FEATURES)
EMBEDDINGS_LEN = 300

embeddings_index = np.zeros((len(vectorizer.get_feature_names()) + 1, EMBEDDINGS_LEN))
for word, idx in word2idx.items():
    try:
        embedding = slp.vocab[word].vector
        embeddings_index[idx] = embedding
    except:
        pass
dl_model = Sequential()
dl_model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                       EMBEDDINGS_LEN,  # Embedding size
                       weights=[embeddings_index],
                       input_length=MAX_SEQ_LENGTH,
                       trainable=False))
dl_model.add(LSTM(300, dropout=0.2))
dl_model.add(Dense(len(set(y)), activation='softmax'))

dl_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
dl_model.fit(X_train_sequences, y_train,
             epochs=5, batch_size=128, verbose=1,
             validation_split=0.1)
scores = dl_model.evaluate(X_test_sequences, y_test, verbose=1)
lstm_prediction = dl_model.predict(X_test_sequences)
print("Deep Learning (LSTM) Accuracy:", scores[1])

# lstm_clf_filename = 'lstm_classifier.pkl'
# pickle.dump(dl_model, open(lstm_clf_filename, 'wb'))
# vectorizer_filename = 'vectorizer.pkl'
# pickle.dump(vectorizer, open(vectorizer_filename, 'wb'))



# print('LSTM Accuracy: ', accuracy_score(X_test_sequences, y_test))
# print(classification_report(y_test, lstm_prediction))
# print('LSTM Accuracy: ', accuracy_score(lstm_prediction, y_test))
# print(classification_report(y_test, lstm_prediction))

