#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: leeyuwen
"""
import sys
import os
import xml.etree.ElementTree as ET
import re
import json
from operator import itemgetter 
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pickle
from nltk.corpus import stopwords, wordnet
import nltk
import argparse
import math

def read_xmlfile(path):
    tree = ET.parse(path)
    root = tree.getroot()
    Article = root.findall("PubmedArticle")
    doc = {}
    for elem in Article:
        article_text = {}
        title = elem.find("MedlineCitation").find("Article").find("ArticleTitle").text
        for article in elem.find("MedlineCitation").find("Article").findall("Abstract"): #內文位置 
            if (article.find('AbstractText').text):
                abs = article.find('AbstractText')
                if 'Label' in abs.attrib:
                    article_text[abs.attrib['Label']] =  abs.text
                else:
                    article_text['Abstract'] =  abs.text
            else:
                continue
        doc[title] = article_text
    del_list = []
    for item in doc.items():
        if item[1] == {}:
            del_list.append(item[0])
    for item in del_list:
        del doc[item]
    
    return doc

def idx_sent(doc):
    idx_article = {}
    for idx, item in enumerate(doc.items()):
        sent_list = []
        split_word = {}
        for article in item[1].items():
            sent_list.append(article[1]) # 只找句子
            text = ' '.join(str(x) for x in sent_list)
            text = text.replace('\n', ' ')
            # split_word[article[1]] = text
        idx_article[item[0]] = text # {文章idx : 文章}

    return idx_article

def split_sent(doc_idx):
    doc_sent = {}
    for idx, item in enumerate(doc_idx.items()):    # {文章: 句子}
        sentences = sent_tokenize(item[1])
        doc_sent[item[0]] = sentences  # {文章 : 切好的句子}

    print(doc_sent)
    return doc_sent

def word_preprocess(doc_sent):
    idx_sent = {}
    idx_word = {}
    voc_set = set()
    for idx, sents in enumerate(doc_sent.items()):
        sent_word = {}
        words = []
        for sent in sents[1]:
            word = text2word(sent) # sent2word
            word = clean_word(word) #stop word
            word = lemma(word) # lemma
            for w in word:
                if w not in voc_set:
                    voc_set.add(w)
            sent_word[sent] = word
            words.append(word)
        text = [str(w) for word in words for w in word]
        idx_sent[sents[0]] = sent_word
        idx_word[idx] = text

    return idx_sent, idx_word, voc_set

'''字串變成單字'''
def text2word(text): 
    words = []
    split_word = text.split(']')
    text = ' '.join(str(x).lower() for x in split_word)
    split_word = text.split("'s")
    for word in split_word:
        words = re.split(r'[!(\')#$"&…%^*+,./{}[;:<=>?@~ \　]+', word)

    return words[:-1]

'''stop word'''
def clean_word(text):
    sentences = [re.sub(r'[^a-z0-9|^-]', ' ', sent.lower()) for sent in text]
    clean_words = []
    for sent in sentences:
        words = [word for word in sent.split() if not word.replace('-', '').isnumeric()]
        words = stop_word(words)
        if(words):
            clean_words.append(' '.join(words))
    return clean_words

'''將sentences 切成 words'''
def splitsent2words(text):
    tokens = [x.split() for x in text]
    return tokens

def porter(words):
    ps = PorterStemmer()
    porter_list = []
    for word in words:
        porter_list.append(ps.stem(word))

    return porter_list

def stop_word(words):
    # nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in words if not w.lower() in stop_words]
    return filtered_sentence

def lemma(sentences):
    # nltk.download('wordnet')
    # nltk.download('averaged_perceptron_tagger')
    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    lemmatizer = WordNetLemmatizer()
    lemma_word = [lemmatizer.lemmatize(sentence, get_wordnet_pos(sentence)) for sentence in sentences]
    return lemma_word



'''字數count'''
def word_count(text):
    article_voc = {}
    for idx, sent in enumerate(text.items()):
        vocabulary = {}
        for words in sent[1].items():
            for word in words[1]:
                if word not in vocabulary:
                    vocabulary[word] = 1
                else:
                    vocabulary[word] += 1
        article_voc[idx] = vocabulary
    print(len(article_voc))
    print(article_voc[0])
    return article_voc

def tf_idf(word, voc_list, idx_word):
    count_sum = 0
    for article in voc_list.items():
        count_sum += article[1] # 字總數
    return tf(word, voc_list, count_sum) * idf(word, idx_word)

def tf(word, voc_list, count_sum): # word count / 總count
    word_count = 0
    if word in voc_list:
        word_count = voc_list[word]
    return word_count / count_sum

'''這個字出現在幾篇文章中'''
def idf(word, idx_word):
    article_count = len(idx_word)
    word_count = 0
    for words in idx_word:
        if word in words:
            word_count += 1
    return math.log(article_count / word_count)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, required=True, help="Filename")
    args = parser.parse_args()
    Filename = args.filename
    path = f'./xml/{Filename}.xml'
    save_path = f'./{Filename}.pickle'
    save_voc = f'./{Filename}_voc.pickle'

    doc = read_xmlfile(path)
    article_idx = idx_sent(doc)
    doc_idx = split_sent(article_idx)
    sents_word, idx_word, voc_set= word_preprocess(doc_idx) # {文章 : {sentence : split sentence}} | {文章 : words}
    voc_list = word_count(sents_word) #{文章 : {word : 出現次數}}
    article_tfidf = {}
    for idx, words in enumerate(idx_word.items()):
        tfidf_list = {}
        for word in words[1]:
            word_tfidf = tf_idf(word, voc_list[idx], idx_word[idx])
            tfidf_list[word] = word_tfidf
        article_tfidf[words[0]] = tfidf_list
    article_top = {}
    for idx, article in enumerate(sents_word.items()):
        sent_top = {}
        for sent in article[1].items():
            if(sent[1]):
                sent_count = 0
                for word in sent[1]:
                    sent_count += article_tfidf[idx][word]
                avg_count = sent_count/len(sent[1])
                sent_top[sent[0]] = avg_count
        
        article_top[article[0]] = sorted(sent_top.items(), key = lambda x : x[1], reverse=True)

    for item in article_top.items():
        print('Title :', item[0])
        for idx, sent in enumerate(item[1][:3]):
            print('Top %s : %s' % (str(idx+1), sent[0]))
        print('')
    
    # with open(save_path, 'wb')as fpick:
    #     pickle.dump(article_top, fpick)
    # with open(save_voc, 'wb')as fpick:
    #     pickle.dump(voc_set, fpick)

if __name__ == '__main__':
    main()