# from keras.preprocessing.text import Tokenizer
import spacy
import csv
import pandas as pd
import sklearn as sk
import math
from sklearn.feature_extraction.text import TfidfVectorizer
nlp = spacy.load("en_core_web_lg")
tf_list = []


def find_title(row, tags):
    lenOfTag = []
    for i in tags:
        word_in_ans = row.lower().split().count(i)
        lenOfTag.append(word_in_ans)
    return lenOfTag


def search_answer(doc):
    # verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    # nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN"]
    nlp.vocab["python"].is_stop = True
    tags = [token.lemma_ for token in doc if token.is_stop == False]
    tags = sorted(set(tags), key=lambda x: tags.index(x))
    print(tags)
    with open('QueryAnswers.csv', mode='r', encoding='UTF8') as f:
        reader = csv.reader(f)
        print("Search Answers:")

        for row in reader:
            words_in_row = len(row[0].split())
            tf_list.append(find_title(row[0], tags))
        # print(tf_list)


def search_question(doc):
    with open('QueryResults_WithAns.csv', mode='r', encoding='UTF8') as f:
        reader = csv.reader(f)

        for row in reader:
            main_doc = nlp(row[1])
            search_doc_no_stop_words = nlp(' '.join([str(t) for t in doc if not t.is_stop]))
            main_doc_no_stop_words = nlp(' '.join([str(t) for t in main_doc if not t.is_stop]))
            word_similarity = search_doc_no_stop_words.similarity(main_doc_no_stop_words)

            if word_similarity >= 0.9:
                print(word_similarity)
                print("Search Questions:")
                print(row)



def ask_question():
    question = input("What is your question? ")
    search_doc = nlp(question)
    search_question(search_doc)
    # search_answer(search_doc)


ask_question()
