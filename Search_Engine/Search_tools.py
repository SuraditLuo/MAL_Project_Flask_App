import csv
import string
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd


# string.punctuation use to remove symbol (e.g. !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~)
def remove_puncts(input_string, string):
    return str(input_string).translate(str.maketrans('', '', string.punctuation)).lower()

def searchByTitle(query):
    data = pd.read_csv('../Resources/Jikan_database.csv')
    title = data['title']
    corpus = title.to_numpy().tolist()
    cleaned_corpus = []
    for doc in corpus:
        cleaned_doc = remove_puncts(doc, string)
        cleaned_corpus.append(cleaned_doc)
    tokenized_clean_corpus = []
    for doc in cleaned_corpus:
        doc = doc.split()
        tokenized_clean_corpus.append(doc)
    bm25 = BM25Okapi(tokenized_clean_corpus)
    relevent_document = 0
    tokenized_query = remove_puncts(query, string).split(" ")
    doc_scores = bm25.get_scores(tokenized_query).tolist()
    for score in doc_scores:
        if score != 0.0:
            relevent_document += 1
    rank = np.argsort(doc_scores)[::-1]
    if (relevent_document > 10):
        return data.iloc[rank[:10]]
    else:
        return data.iloc[rank[:relevent_document]]

def searchByDescription(query):
    data = pd.read_csv('../Resources/Jikan_database.csv')
    title = data['synopsis']
    corpus = title.to_numpy().tolist()
    cleaned_corpus = []
    for doc in corpus:
        cleaned_doc = remove_puncts(doc, string)
        cleaned_corpus.append(cleaned_doc)
    tokenized_clean_corpus = []
    for doc in cleaned_corpus:
        doc = doc.split()
        tokenized_clean_corpus.append(doc)
    bm25 = BM25Okapi(tokenized_clean_corpus)
    relevent_document = 0
    tokenized_query = remove_puncts(query, string).split(" ")
    doc_scores = bm25.get_scores(tokenized_query).tolist()
    for score in doc_scores:
        if score != 0.0:
            relevent_document += 1
    rank = np.argsort(doc_scores)[::-1]
    if (relevent_document > 10):
        return data.iloc[rank[:10]]
    else:
        return data.iloc[rank[:relevent_document]]