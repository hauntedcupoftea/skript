
# *skript*

A repository for a Probability and Statistics Hackathon conducted at Bennett University during 5-6th November, 2022.

## IDEA

The main idea of the project is to take a user input in the form of a speech transcript/article, and then use basic statistical analysis to grade the speech on a spider chart consisting of 5 axes -

- information
- persuasion
- description
- entertainment
- ceremony/tribute (special occasions or outliers)
  
A seperate component (targeted) that uses KNN to find out what the speech is closest to, with an existing dataset of some famous speeches and addresses.

## APPROACH

The general approach to achieve such statistical data and present it involves using Natural Language Processing techniques like tokenization, cosine similarity, normalization of data against a large enough dataset. The model being used to create the dataset that the input data will be checked against is [Spacy](https://spacy.io). Basic text cleanup techniques are being used, like regex filtering and removing stop-words.

The functionality that we plan to add include getting a simple analysis of just one input (either directly through clipboard/keyboard or a .txt file), and a future functionality of adding multiple drafts of the same documents and then comparing their charts to find the one best suited for the application.

Another basic functionality includes a simple bayesian classifier that attempts to predict the overall dominating type of the speech/script. We hypothesize this might be the same as the quality with the highest score on the spider chart, but we do not know for sure. Little to no background research can be found on this.
