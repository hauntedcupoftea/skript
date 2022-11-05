import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import re
import tensorflow as tf
import spacy
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

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
            cleanwords.append(x.lower())
    return (cleanwords, cleansample)

