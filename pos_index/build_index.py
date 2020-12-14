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
import pickle

# create directory for saving text information from transcripts
dir = 'ECTText'
path = os.path.join(os.getcwd(), dir)

# stopwords = code for downloading stop words through nltk
stopword = stopwords.words('english')

# apply lemmatization
wordnet_lemmatizer = WordNetLemmatizer()

InvPosIdx = dict()

for i, filename in enumerate(os.listdir(path)):

    if i % 100 == 0:
        print(i)

    corpus = nltk.data.load(os.path.join(path, filename), format='raw').decode('utf-8')

    # remove punctuations
    corpus = re.sub(r'[^\w\s]', ' ', corpus)
    # print(corpus)

    tokens = []
    for sentences in re.split("\n", corpus):
        curr_tokens = word_tokenize(sentences, language='english')
        curr_tokens = [token.lower() for token in curr_tokens if token not in stopword and token.isalnum()]
        tokens.extend([wordnet_lemmatizer.lemmatize(token) for token in curr_tokens])

    for token in set(tokens):
        if token not in InvPosIdx.keys():
            InvPosIdx[token] = []
        positions = [i for i, c_token in enumerate(tokens) if c_token == token]
        InvPosIdx[token].append((filename.split('-')[0], positions))


with open(os.path.join(os.getcwd(), 'InvPosIdx.pkl'), 'wb') as f:
    pickle.dump(InvPosIdx, f, pickle.HIGHEST_PROTOCOL)
