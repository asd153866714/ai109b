import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
count = CountVectorizer()

data = pd.read_csv(
    "./Test.csv")
print(data.head())

# 資料的預處理


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emojis).replace('-', '')
    return text


data['text'] = data['text'].apply(preprocessor)

# split the sentences into words => 詞幹提取

porter = PorterStemmer()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# TF-IDF features
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None,
                        tokenizer=tokenizer_porter, use_idf=True, norm='l2', smooth_idf=True)

# Learn vocabulary and idf, return term-document matrix.This is equivalent to fit followed by transform, but more efficiently implemented.
y = data.label.values
x = tfidf.fit_transform(data.text)

# split dataset into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    x, y, random_state=1, test_size=0.5, shuffle=False)

# using Logistic Regression for our Sentiment Analysis
clf = LogisticRegressionCV(cv=6, scoring='accuracy', random_state=0,
                           n_jobs=-1, verbose=3, max_iter=500).fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
