import pandas as pd
import re
from html import unescape
import string

import ssl
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer


def loadDataSet(filename):
    dataset = pd.read_csv(filename)
    dataset = dataset.sample(frac=1)
    return dataset

def removeExtra(dataset):
    for key, row in dataset.iterrows():
        row['Description'] = re.sub(r'https?:\/\/.\S+', "", row['Description'])
        row['Description'] = re.sub(r'www.\S+', "", row['Description'])
        row['Description'] = re.sub(r'#', '', row['Description'])
        row['Description'] = re.sub(r'\d+', '', row['Description'])
        row['Description'] = unescape(row['Description'])
    return dataset

def makeLowercase(dataset):
    for key, row in dataset.iterrows():
        row['Description'] = row['Description'].lower()
    return dataset


def replaceContractions(dataset):
    Apos_dict = {"'s": " is", "n't": " not", "'m": " am", "'ll": " will",
                 "'d": " would", "'ve": " have", "'re": " are"}

    for ids, row in dataset.iterrows():
        for key, value in Apos_dict.items():
            if key in row['Description']:
                row['Description'] = row['Description'].replace(key, value)

        for key, row in dataset.iterrows():
            # I don't understand why this line is needed. Will leave in until I can ask.
            row['Description'] = " ".join([s for s in re.split("([A-Z][a-z]+[^A-Z]*)", row['Description']) if s])
        return dataset

def removeStopwords(dataset):
    stopwords_eng = stopwords.words('english')
    for key, row in dataset.iterrows():
        row['Description'] = row['Description'].split()
        token_list_d = []
        for word in row['Description']:
            if (word not in stopwords_eng) and (word not in string.punctuation):
                word = word.translate(str.maketrans('', '', string.punctuation))
                token_list_d.append(word)
        row['Description'] = token_list_d
    return dataset

def lemmatizeWords(dataset):
    wnl = WordNetLemmatizer()
    for key, row in dataset.iterrows():
        stemmer = PorterStemmer()
        for word in row["Description"]:
            stemmer.stem(word)
            wnl.lemmatize(word)
        row['Description'] = TreebankWordDetokenizer().detokenize(row['Description'])
    return dataset

def preprocess(filename):
    dataset = loadDataSet("trainingData.csv")
    dataset = removeExtra(dataset)
    dataset = makeLowercase(dataset)
    dataset = replaceContractions(dataset)
    dataset = removeStopwords(dataset)
    dataset = lemmatizeWords(dataset)
    return dataset