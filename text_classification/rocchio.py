from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

import nltk
import nltk.data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

import re
import os
import sys
import math
import heapq
import pickle
import numpy as np
from bs4 import BeautifulSoup
from collections import Counter
import pandas as pd

# stopwords = code for downloading stop words through nltk
stopword = stopwords.words('english')

# apply lemmatization
wordnet_lemmatizer = WordNetLemmatizer()


def custom_tokenizer(sentences):

    # remove punctuations
    sentences = re.sub(r'[^\w\s]', ' ', sentences)

    tokens = word_tokenize(sentences, language='english')
    tokens = [token.lower() for token in tokens if token not in stopword and token.isalnum() and len(token) > 1]
    tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    return tokens

if __name__ == "__main__":

    # create directory for saving text information from transcripts
    path = sys.argv[1]

    path_class_1_train = path + 'class1/train/'
    path_class_2_train = path + 'class2/train/'

    class_1_file_count_train = len(os.listdir(path_class_1_train))
    class_1_file_name_train = sorted(os.listdir(path_class_1_train), key=lambda x: int(x.split('.')[0]))
    class_1_file_name_train = [os.path.join(path_class_1_train, filename) for filename in class_1_file_name_train]

    class_2_file_count_train = len(os.listdir(path_class_2_train))
    class_2_file_name_train = sorted(os.listdir(path_class_2_train), key=lambda x: int(x.split('.')[0]))
    class_2_file_name_train = [os.path.join(path_class_2_train, filename) for filename in class_2_file_name_train]

    corpus_train = class_1_file_name_train + class_2_file_name_train
    Y_train = [0] * len(class_1_file_name_train) + [1] * len(class_2_file_name_train)

    path_class_1_test = path + 'class1/test/'
    path_class_2_test = path + 'class2/test/'

    class_1_file_count_test = len(os.listdir(path_class_1_test))
    class_1_file_name_test = sorted(os.listdir(path_class_1_test), key=lambda x: int(x.split('.')[0]))
    class_1_file_name_test = [os.path.join(path_class_1_test, filename) for filename in class_1_file_name_test]

    class_2_file_count_test = len(os.listdir(path_class_2_test))
    class_2_file_name_test = sorted(os.listdir(path_class_2_test), key=lambda x: int(x.split('.')[0]))
    class_2_file_name_test = [os.path.join(path_class_2_test, filename) for filename in class_2_file_name_test]

    corpus_test = class_1_file_name_test + class_2_file_name_test
    Y_test = [0] * len(class_1_file_name_test) + [1] * len(class_2_file_name_test)

    vectorizer_train = CountVectorizer(
        input='filename',
        analyzer='word',
        strip_accents='unicode',
        decode_error='ignore',
        tokenizer=custom_tokenizer,
    )

    X_train = vectorizer_train.fit_transform(corpus_train)
    X_test = vectorizer_train.transform(corpus_test)

    tfidfTransform = TfidfTransformer()
    tfidf_train = tfidfTransform.fit_transform(X_train.toarray())
    tfidf_test = tfidfTransform.transform(X_test.toarray())
    mean_class_1 = np.mean(tfidf_train[0:len(class_1_file_name_train)], axis=0)
    mean_class_2 = np.mean(tfidf_train[len(class_1_file_name_train):], axis=0)

    b = 0
    Y_pred = (np.linalg.norm(tfidf_test - mean_class_2, axis=1) < np.linalg.norm(tfidf_test - mean_class_1, axis=1)).astype(int)
    with open(sys.argv[2], 'w') as f:
        f.write(f'Output File 2\n')
        f.write(f"b \t 0\n")
        f.write(f"Rocchio {f1_score(Y_test, Y_pred, average='macro')}\n")
