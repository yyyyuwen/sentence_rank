from logging import debug
from flask import Flask, render_template, request, redirect, send_from_directory, Response
import os
import xml.etree.ElementTree as ET
import re
import json
from operator import itemgetter 
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import difflib
import pickle
import torch
from scipy.special import softmax
from markupsafe import Markup
from sklearn.decomposition import PCA
import itertools

IMG_FOLDER = os.path.join('static', 'img')

app = Flask(__name__)
app.config['IMG_FOLDER'] = IMG_FOLDER
TEXT = []
app.config['TEXT'] = TEXT
ROUTE = ''
app.config['ROUTE'] = ROUTE

@app.route('/')
def Home():
    with open('./covid.pickle','rb') as file:
        covid = pickle.load(file)
    with open('./virus.pickle','rb') as file:
        virus = pickle.load(file)
    with open('./covid_tfidf','rb') as file:
        covid_tfidf = pickle.load(file)
    with open('./virus_tfidf','rb') as file:
        virus_tfidf = pickle.load(file)
    
    return render_template('index.html', **locals(), enumerate = enumerate)


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port='8080')