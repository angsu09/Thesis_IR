import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
title = []
cleaned_ques = []
keywords = []
vectorizer = TfidfVectorizer(
    stop_words="english"
)
def tf_idf_Categories():

    with open('Corpus2.0.csv', mode='r', encoding='UTF8') as f:
        next(f)
        reader = csv.reader(f)
        for row in reader:
            cleaned_ques.append(row[2])
            title.append(row[0])
        vectors = vectorizer.fit_transform(cleaned_ques)
        feature_names = np.array(vectorizer.get_feature_names())
        n = 10
        dense = vectors.todense()
        denselist = dense.tolist()
        denseArr = np.array(dense)
        for i in denseArr:
            tfidf_sorting = np.argsort(i).flatten()[::-1]
            top_n = feature_names[tfidf_sorting][:n]
            keywords.append(top_n)
        # add_keyword(keywords)
        return vectors

def K_means():
    Sum_of_squared_distances = []
    K = range(2, 10)
    for k in K:
        km = KMeans(n_clusters=k, max_iter=200, n_init=10)
        km = km.fit(categories_tfidf)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

def find_cluster_posts():
    true_k = 7
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=500, n_init=10)
    model.fit(categories_tfidf)
    labels = model.labels_
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 4)
    pd.set_option('display.width', 2000)
    wiki_cl = pd.DataFrame(list(zip(title, keywords, labels)), columns=['title', 'keywords', 'cluster'])
    # wiki_cl = pd.DataFrame(labels)
    print(wiki_cl.sort_values(by=['cluster']))
    print(wiki_cl)

categories_tfidf = tf_idf_Categories()
K_means()