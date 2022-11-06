import numpy as np
import pandas as pd
from scipy.spatial import distance
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
    # deprecated script to seperate all words
    # words = [i.split() for i in cleansample]
    # punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    # cleanwords = []
    # for i in words:
    #     for x in i:
    #         x.replace(punc, "")
    #         if [pog.is_stop for pog in nlp(x)] == [False]:
    #             cleanwords.append(x.lower())
    return cleansample

def attrSep(proc: spacy.tokens.doc.Doc) -> list:
    info = proc.similarity(nlp('information facts assertion'))
    pers = proc.similarity(nlp('imperative persuasive argument'))
    desc = proc.similarity(nlp('descriptive demonstration workshop'))
    entr = proc.similarity(nlp('entertaining witty engaging'))
    cerm = proc.similarity(nlp('occasion ceremony tribute'))
    return [info, pers, desc, entr, cerm]

def NearestNeighbour(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return closest_index

def analyse(data: list) -> list:
    reslist = []
    matchData = pd.read_csv('defdata.csv')
    maxd = matchData.max().tolist()
    mind = matchData.min().tolist()

    for i in range(len(data)):
        cleansample = clean(str(data[i]))
        rawspeech = " ".join(cleansample.tolist())
        proc = nlp(rawspeech)
        attrList = attrSep(proc)
        scaledattr = [max(0.1, ((attrList[x] - mind[x+3])/(maxd[x+3] - mind[x+3]))) for x in range(5)]
        close = NearestNeighbour(attrList, matchData.loc[:, 'informative':'ceremonial'])
        reslist.append(scaledattr + matchData.loc[close, 'title':'classification'].tolist())

    return reslist

if __name__ == '__main__':
    text = ["The FitnessGram Pacer Test is a multistage aerobic capacity test that progressively gets more difficult as it continues. The 20 meter pacer test will begin in 30 seconds. Line up at the start. The running speed starts slowly but gets faster each minute after you hear this signal bodeboop. A sing lap should be completed every time you hear this sound. ding Remember to run in a straight line and run as long as possible. The second time you fail to complete a lap before the sound, your test is over. The test will begin on the word start. On your mark. Get ready!… Start. ding",
            "Just watched slender man. Now this might sound crazy but all my life I’ve watched documentaries played the game and now watched the movie about slender man. To be honest I believe he is real. There are many sightings many unexplained disappearances and the one thing that gets to me all the time in every myth or legend about him he is explained exactly the same way. A thin man like figure with no face. And nearly if not all video sightings and images of him look extremely alike. Also the thing that bugs me the most is that people that claim to have crossed paths with him explain him and draw him the same way. Now some people might argue and say that’s cause they’ve heard about him. Well I say this where did it all start. It can’t be just a rumour because rumours fade away with time. Can’t be a myth cause too many people claim to have seen him. Can’t be legend because he is supposedly still around. And one more thing. There’s not just one because he’s been sighted all around the world. Fact is if he is real the smartest thing to do is to not look for him or try contacting him because that’s what draws him to you. He plays mind games making people go insane or question what is reality and when he’s got you accepting that he is real that’s when he comes for you. Now the only thing I question about his existence is weather or not he is really a demon. All I know is that I believe he is real and that he’s not someone to fuck with. I didn’t once flinch watching this movie and usually I laugh at horror but this time I just sat and watched. Documentary’s of him say that he only shows himself if you go looking for him. Certain religions that believe in him say he’s the embodiement of the angel of death. Others say he is death. Now if you actually read this whole status and have anything to add on what you know I’m all ears because to be honest he doesn’t scare me. I’m intrigued to know why he does what he is and how he does what he does. Research suggests that the only way to fight back is to let go of all your fears including death itself. So to be honest at the stage of my life now if I ever come face to face with him I hope actually I know that I will be one of the ones to survive him and learn the truth"]
    result = analyse(text)
    print(result)