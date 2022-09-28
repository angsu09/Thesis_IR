from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import re

from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from tabulate import tabulate
stops = set(stopwords.words("english"))
not_stopwords = {'if'}
final_stop_words = set([word for word in stops if word not in not_stopwords])
def pre_process(text):
    # lowercase
    text = text.lower()

    # remove tags
    text = re.sub("", "", text)

    # remove special characters and digits
    text = re.sub("[^a-zA-Z0-9]+", " ", text)

    return text


main_doc = pd.read_csv('Corpus2.0.csv')
main_doc['Title + Body'] = main_doc['Title + Body'].apply(lambda x: pre_process(x))

doc = main_doc['Title + Body'].tolist()
cv = CountVectorizer(max_df=0.85, stop_words=final_stop_words)
word_count_vector = cv.fit_transform(doc)
print(word_count_vector.shape)

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)

def findClusterVec():
    cluster_doc = pd.read_csv("Clusters.csv")
    cluster_vec = tfidf_transformer.transform(cv.transform(cluster_doc["Cluster tags"]))
    # print("Cluster vec:\n\n", cluster_vec)
    # print()
    # print(cluster_vec.shape)
    return cluster_vec


def customise_CS_Result(cs):
    res = []
    for i in range(len(cs)):
        key = {"id": i, "key": cs[i]}
        if cs[i] > 0:
            res.append(key)
    res = sorted(res, key=lambda x: x['key'], reverse=True)
    return res


def buildQuesTags(ques):
    clean_ques = pre_process(ques)
    # print(clean_ques)
    ques_vec = tfidf_transformer.transform(cv.transform([clean_ques]))
    return ques_vec


def findQuesCluster(ques_vec, cluster_vec):
    cos = cosine_similarity(ques_vec, cluster_vec).flatten()
    res = customise_CS_Result(cos)
    print(res)
    if res != []:
        print("Cluster: \t", res[0]["id"])
        return res[0]["id"]



def findMatchQuestion(clus, ques_vec):
    row_details = []
    for row in main_doc.iterrows():
        if int(clus) == row[1]["Cluster"]:
            row_details.append(row[1])
    row_df = pd.DataFrame(row_details)
    row_df['Title + Body'] = row_df['Title + Body'].apply(lambda x: pre_process(x))
    doc = row_df['Title + Body'].tolist()
    doc_vec = tfidf_transformer.transform(cv.transform(doc))

    main_cos = cosine_similarity(ques_vec, doc_vec).flatten()
    c = zip(main_cos.tolist(), row_df['Title'].tolist())
    cos_df = pd.DataFrame(c, columns=['cosine', 'title'])
    # cos_df = cos_df.sort_values(by=['cosine'], ascending=False)
    cos_df = cos_df.nlargest(n=10, columns=["cosine"])
    print("Cosine: \n", cos_df.to_string(index=False))

def askQuestion():
    question = input("What is your question?")

    ques_vec = buildQuesTags(question)

    cluster_vec = findClusterVec()
    cluster = findQuesCluster(ques_vec, cluster_vec)

    if cluster is not None:
        findMatchQuestion(cluster, ques_vec)

    else:
        print("Cluster is not found")
        print()
    askQuestion()

askQuestion()
