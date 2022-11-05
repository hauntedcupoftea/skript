import numpy as np
import pandas as pd
import re
import tensorflow as tf
import spacy
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
spacy.prefer_gpu()
nlp = spacy.load('en_core_web_lg')

def clean(text):
    sample = np.array(re.split(r'(\. )|(\n)', text))
    sample = sample[sample != np.array(None)]
    cleansample = np.array([i for i in sample if len(i) > 6])
    words = [i.split() for i in cleansample]
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    cleanwords = []
    for i in words:
        for x in i:
            x.replace(punc, "")
            if [pog.is_stop for pog in nlp(x)] == [False]:
                cleanwords.append(x.lower())
    return cleansample

def attrSep(proc: spacy.tokens.doc.Doc) -> list:
    info = proc.similarity(nlp('information facts assertion'))
    pers = proc.similarity(nlp('imperative persuasive argument'))
    desc = proc.similarity(nlp('descriptive demonstration workshop'))
    entr = proc.similarity(nlp('entertaining witty engaging'))
    cerm = proc.similarity(nlp('occasion ceremony tribute'))
    return [info, pers, desc, entr, cerm]

def analyse(data: list):
    reslist = []
    matchData = pd.read_csv('defdata.csv')
    maxd = matchData.max().tolist()
    mind = matchData.min().tolist()
    for i in data:
        cleansample = clean(i)
        rawspeech = " ".join(cleansample.tolist())
        proc = nlp(rawspeech)
        attrList = attrSep(proc)
        scaledattr = [((attrList[x] - mind[x])/(maxd[x] - mind[x])) for x in range(5)]
        reslist.append(scaledattr)
    return reslist
