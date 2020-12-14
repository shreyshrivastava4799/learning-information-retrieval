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


class Token():

    def __init__(self, file_count):
        super(Token, self).__init__()
        self.idf = 0
        self.df = np.zeros(file_count)
        self.posting_list = []


def tf_idf_score(query, doc_tf_idf):

    doc_tf_idf = (doc_tf_idf @ query) / (np.linalg.norm(doc_tf_idf, axis=1) * np.linalg.norm(query))
    index = heapq.nlargest(10, range(len(doc_tf_idf)), doc_tf_idf.take)

    return [(idx, doc_tf_idf[idx]) for idx in index]


def local_Champion_List_Score(q_tokens, doc_tf_idf, filesCLLocal):

    doc_tf_idf = doc_tf_idf[filesCLLocal, :]
    doc_tf_idf = (doc_tf_idf @ query) / (np.linalg.norm(doc_tf_idf, axis=1) * np.linalg.norm(query))
    index = heapq.nlargest(10, range(len(doc_tf_idf)), doc_tf_idf.take)

    return [(filesCLLocal[idx], doc_tf_idf[idx]) for idx in index]


def global_Champion_List_Score(query, doc_tf_idf, filesCLGlobal):

    doc_tf_idf = doc_tf_idf[filesCLGlobal, :]
    doc_tf_idf = (doc_tf_idf @ query) / (np.linalg.norm(doc_tf_idf, axis=1) * np.linalg.norm(query))
    index = heapq.nlargest(10, range(len(doc_tf_idf)), doc_tf_idf.take)

    return [(filesCLGlobal[idx], doc_tf_idf[idx]) for idx in index]


def cluster_Prunning_Score(query, leader_tf_idf, doc_tf_idf, followers):

    # for each token in the vocab we have its inverse of document frequency
    # also we have for each token the term frequency in a document
    # now we want to calculate tf_idf(matching) score corresponding to a combination of query tokens for only leader documents
    # for this :
    #   we first find the tf_idf of each doc. and multiply with doc mask
    #   next we take product with query vec and find doc with max match
    #   we take its followers tf_idf score and then again product query vec with them to finally find the most relevant doc

    closest_leader = np.argmax((leader_tf_idf @ query) / (np.linalg.norm(leader_tf_idf, axis=1) * np.linalg.norm(query)))
    follower_tf_idf = doc_tf_idf[followers[closest_leader], :]
    doc_tf_idf = follower_tf_idf @ query
    index = heapq.nlargest(10, range(len(doc_tf_idf)), doc_tf_idf.take)
    return [(followers[closest_leader][idx], doc_tf_idf[idx]) for idx in index]


if __name__ == "__main__":

    vocab = dict()

    # create directory for saving text information from transcripts
    path = '../Dataset'

    # stopwords = code for downloading stop words through nltk
    stopword = stopwords.words('english')

    # apply lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()

    file_count = len(os.listdir(path))
    file_name = sorted(os.listdir(path), key=lambda x: int(x.split('.')[0]))

    for i, filename in enumerate(file_name):

        corpus = nltk.data.load(os.path.join(path, filename), format='raw').decode('utf-8')
        soup = BeautifulSoup(corpus, 'html.parser')

        # remove punctuations
        corpus = re.sub(r'[^\w\s]', ' ', soup.get_text())

        tokens = []
        for sentences in re.split("\n", corpus):
            curr_tokens = word_tokenize(sentences, language='english')
            curr_tokens = [token.lower() for token in curr_tokens if token not in stopword and token.isalnum() and len(token) > 1]
            tokens.extend([wordnet_lemmatizer.lemmatize(token) for token in curr_tokens])
        counter_tokens = Counter(tokens)

        for token, freq in counter_tokens.items():
            if token not in vocab.keys():
                vocab[token] = Token(file_count)
            vocab[token].posting_list.append((filename, math.log10(1 + freq)))
            vocab[token].df[i] = math.log10(1 + freq)

    # calc Inverse Positional Index
    InvPosIndex = dict()
    for token, value in vocab.items():
        idf = math.log10(file_count / len(value.posting_list))
        vocab[token].idf = idf
        InvPosIndex[(token, idf)] = value.posting_list

    # calc Champion List Local
    CLLocal = dict()
    for token, value in vocab.items():
        CLLocal[token] = sorted(value.posting_list, key=lambda x: x[1], reverse=True)[:50]
        CLLocal[token] = [int(tup[0].split('.')[0]) for tup in CLLocal[token]]

    with open('../StaticQualityScore.pkl', 'rb') as f:
        data = pickle.load(f)

    # calc Champion List Global
    CLGlobal = dict()
    for token, value in vocab.items():
        CLGlobal[token] = sorted(value.posting_list, key=lambda x: x[1] * vocab[token].idf + data[int(x[0].split('.')[0])], reverse=True)[:50]
        CLGlobal[token] = [int(tup[0].split('.')[0]) for tup in CLGlobal[token]]

    outfile = open(os.path.join(os.getcwd(), 'RESULTS2_17CS30034.txt'), 'w')

    with open('../Leaders.pkl', 'rb') as f:
        Leaders = pickle.load(f)

    # tf_idf score for all the documents
    doc_df = np.zeros((len(vocab), file_count))
    for i, value in enumerate(vocab.values()):
        doc_df[i, :] = value.df
    doc_tf_idf = doc_df.transpose() * np.array([value.idf for value in vocab.values()])

    # tf_idf score for leader documents
    leader_tf_idf = doc_tf_idf[Leaders, :]

    # find the followers for the leaders
    followers = [[] for i in range(len(Leaders))]
    for i in range(doc_tf_idf.shape[0]):
        index = np.argmax(np.sum((doc_tf_idf[i] * leader_tf_idf), axis=1) /
                          ((np.linalg.norm(doc_tf_idf[i]) * np.linalg.norm(leader_tf_idf, axis=1))))
        followers[index].append(i)

    # read the file provided as an argument
    with open(sys.argv[1]) as fp:

        for line in fp:
            q_word = line.strip()

            q_word = re.sub(r'[^\w\s]|[0-9]', ' ', q_word)

            q_tokens = word_tokenize(q_word, language='english')
            q_tokens = [token.lower() for token in q_tokens if token not in stopword and token.isalnum() and len(token) > 1]
            q_tokens = [wordnet_lemmatizer.lemmatize(token) for token in q_tokens]

            # crete a vector for query using just idf correponding to query tokens
            query = np.array([(token in q_tokens) * vocab[token].idf for token in vocab.keys()])

            text = line
            scores = tf_idf_score(query, doc_tf_idf)
            for file, score in scores:
                text += f' <{file_name[file]}, {score}>;'
            text += '\n'

            filesCLLocal = []
            for token in q_tokens:
                filesCLLocal.extend(CLLocal[token])
            filesCLLocal = list(set(filesCLLocal))

            filesCLGlobal = []
            for token in q_tokens:
                filesCLGlobal.extend(CLGlobal[token])
            filesCLGlobal = list(set(filesCLGlobal))

            scores = local_Champion_List_Score(query, doc_tf_idf, filesCLLocal)
            for file, score in scores:
                text += f' <{file_name[file]}, {score}>;'
            text += '\n'

            scores = global_Champion_List_Score(query, doc_tf_idf, filesCLGlobal)
            for file, score in scores:
                text += f' <{file_name[file]}, {score}>;'
            text += '\n'

            scores = cluster_Prunning_Score(query, leader_tf_idf, doc_tf_idf, followers)
            for file, score in scores:
                text += f' <{file_name[file]}, {score}>;'
            text += '\n'

            outfile.write(text)

    outfile.close()
