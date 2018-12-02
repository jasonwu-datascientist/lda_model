from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models, matutils
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict
import pandas as pd
import nltk
from nltk.corpus import stopwords
import warnings
import spacy

from spacy.lang.en import English
import gensim
import pyLDAvis.gensim
import pickle
import matplotlib.pyplot as plt
import io
import base64
import matplotlib.patches as patches

#Words to be added to the new stopwords list
new_stopwords=['photo','photos','video','videos','new','day','australia','man','2018','said','one','also','ms','mr','year','people','nbsp','br','say','http','www','href','com']
updated_stop=stopwords.words('english')
for i in new_stopwords:
    updated_stop.append(i)

#Using regular expression to retain only tokens with letters of any case and between a length of 5-9 characters
regexp='[a-zA-Z]{5,9}'

topics_labels = {
   0: "Politics",
   1: "Culture",
   2: "Crime",
    3: "Environment"
}

dictionary = gensim.corpora.Dictionary.load('../3. Models/Articles/dictionary_article.gensim')
corpus = pickle.load(open('../3. Models/Articles/corpus_article.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('../3. Models/Articles/articles_model4.gensim')

parser = English()
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN

def lemmatize_word(words):
    lemma=WordNetLemmatizer()
    lemma_list=[]
    words = word_tokenize(words)
    pos=nltk.pos_tag(words)
    for i,w in enumerate(words):
        meaning=get_wordnet_pos(str(pos[i][1]))
        lemma_list.append(lemma.lemmatize(w,meaning))
    return (" ".join(lemma_list))

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 5]
    tokens = [token for token in tokens if token not in updated_stop]
    tokens = [lemmatize_word(token) for token in tokens]

    return tokens


def predict_topic(text):
    tokens = prepare_text_for_lda(text)
    example=dictionary.doc2bow(tokens)
    a=0
    b=0
    for i in lda.get_document_topics(example):

        if i[1]>a:
            a=i[1]
            b=i[0]

    return (topics_labels[b],a)
    print('Prediction:',topics_labels[b],'\nProbability:',a)
    print('Overall spread of probability:',lda.get_document_topics(example))
    print('Topic labels:',topics_labels)