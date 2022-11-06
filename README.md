
# *skript*

A project for a Probability and Statistics Hackathon conducted at Bennett University during 5-6th November, 2022. Created by Team Number 16 at the hackathon itself, consisting of members Aayush Gupta, Anand Chauhan, and Vasu Jain. The entire project was developed from scratch over the course of 18 hours.

## `IDEA`

The main idea of the project is to take a user input in the form of a speech transcript/article, and then use basic statistical analysis to grade the speech on a spider chart consisting of 5 axes -

- information
- persuasion
- description
- entertainment
- ceremony/tribute (special occasions or outliers)
  
A seperate component has been implemented that uses euclidean distance to find out what the speech is closest to, with an existing dataset of 75 famous speeches and addresses, as a fun side-functionality.

## `APPROACH`

The general approach to achieve such statistical data and present it involves using Natural Language Processing techniques like tokenization, cosine similarity, normalization of data against a large enough dataset. The model being used to create the dataset that the input data will be checked against is [Spacy](https://spacy.io). Basic text cleanup techniques are being used, like regex filtering and removing stop-words.

The functionality that have added includes getting a simple analysis of just one input (either directly through clipboard/keyboard or a .txt file), and a future prospect to implement adding multiple drafts of the same documents and then comparing their charts to find the one best suited for the application.

Basic functionality includes a simple cosine similarity script that attempts to grasp the similarity of the speech to a subset of sentiments that we found. This lets us chart the input speech along with our custom-made dataset, and then allows for comparisons like geometric and vector calculations, which in turn lets us find the closest neighbour to the input speech as a fun side function.

Based on this, we could provide an overall educated prediction about what overall tone and type of the speech is.

All the work on the dataset was done with Vasu Jain, while all the statistics and mathematics behind the project was handled by Anand Chauhan. Aayush Gupta worked on the UI and presentation of the project.

## `PRESENTATION AND USAGE`

The working part of the project only requires three files -

- [The custom dataset](defdata.csv)
- [The Analyser file](skriptAnalyser.py)
- [The UI file](placeholderURL) (file needed)

The UI file uses [Google colaboratory](https://colab.research.google.com/notebooks/widgets.ipynb) widgets created by Aayush Gupta. As such, the UI side of the project will only work on google colaboratory. If you wish to run the project like that, go into a session, open the UI file, then upload the analyser file and the dataset seperately. You will also need to install the specific model of spacy used for the project, but the UI file already has a script for that.

The UI itself is fairly intuitive and user-friendly, and will provide the output in the form of spider charts. Multiple inputs can be plotted at once, although comparison betweent them has not been automated just yet.

Otherwise, only the analyser file and the dataset can be used as modules to be added to a project. Bear in mind however, that the analyser file accepts input as a `python list` only. even if the input is just one text file, it must be wrapped in a list first.

## `THE OTHER FILES`

Some information about the other files:

- [Experimentation Notebook](skriptExperimentation.ipynb): This file contains all the ramblings and experimentation done over the course of the entire project. it also contains the code that converted our raw speech dataset into the plotting dataset we could use for comparison.
- [Sample](sample.txt): A basic sample speech used to get familiar with the workings of all the libraries, as a vision for what the final project would look like was devised.
