import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from numpy import *
import re
from scipy.sparse import csr_matrix
from ast import literal_eval
from scipy.sparse import coo_matrix, vstack
# sentence = nltk.word_tokenize('looks')
# sentence_lemma = []
#
# lemma = WordNetLemmatizer()
#
# pos_tag = nltk.pos_tag(sentence)
# print pos_tag
# for tag in pos_tag:
#     if tag[1]=='NNS':
#         print tag[0]
#         sentence_lemma.append(lemma.lemmatize(tag[0]))
#     else:
#         sentence_lemma.append(tag[0])
# sentence_lemma = ' '.join(sentence_lemma)
# print sentence_lemma

# def lemmatization_nouns(sents):
#     sentence_lemma = []
#     lemma = WordNetLemmatizer()
#     sentence_tag = []
#     a_list = 1
#     if type(sents)!=list:
#         a_list = 0
#         sents = nltk.word_tokenize(sents)
#     for word in sents:
#         sentence_tag.extend(nltk.pos_tag([word]))
#     for tag in sentence_tag:
#         if tag[1]=='NNS':
#             sentence_lemma.append(lemma.lemmatize(tag[0]))
#         else:
#             sentence_lemma.append(tag[0])
#     if not a_list:
#         sentence_lemma = ' '.join(sentence_lemma)
#     return  sentence_lemma
# #print lemmatization_nouns('having used it for more than a month , i can say that it consistently produces topnotch photos')
# print lemmatization_nouns(['photos','cats'])
# #print lemmatization_nouns('i am a good cats')
# a = ['a', 'b', 'c', 'd', 'e', 'a']
# print np.random.choice(a, int(len(a)*0.5), replace=False)

# b = np.array([])
# for i in range(3):
#     b = np.append(b,1)
# print b
# d = np.array([[6,7,3,6],[1,2,3,4]])
#c = np.row_stack((a, b))
#a = np.insert(a,len(a)+1, values=b, axis=0)
#
# a = np.array([])
# def matrix(b):
#     global a
#     for item in b:
#         if a.size == 0:
#             a = np.append(a, item)
#         else:
#             a = np.row_stack((a, item))
#     print a
#
# matrix(b)
# matrix(d)
# a = np.array([[0,2,1,0],[0,0,0,2],[0,0,0,4],[0,0,1,2],[0,0,2,0]])
# b = np.array([0,0,0,0])
# A = coo_matrix(a)
# B = coo_matrix(b)
# c = coo_matrix((1, 3*2+1), dtype=np.int8)
# print c.todense()

#print vstack([cooA, cooB]).todense()
# vstack([cooA,cooB])
# def sparse_matrix(cooA, cooB):
#     global c
#     if len(c.todense()) == 1:
#         c = vstack([cooA, cooB])
#     else:
#         c = vstack([c, cooB])
#     return c
# sparse_matrix(A, B)
# result = sparse_matrix(A, B)
# rint len(result.todense())
# a = 'it is a good feature'
# c = (1, 4)
# add_notation = a.split()
# add_notation[min(c)] = '<e1>'+add_notation[min(c)]+'</e1>'
# add_notation[max(c)] = '<e2>'+add_notation[min(c)]+'</e2>'
# print ((' ').join(add_notation)
# unicodestring = u"Hello world"
# utf8string = unicodestring.encode("utf-8")
# a = '(13,2,3)'
# print tuple(a)
# dt = (1,2,3,4)
# b = str(dt)
# print type(b)
# a = 'i am a appale, and a banana'
# sentence_list = re.findall(r'\w+\-?\w+\-?\w+|\'?\w+', a)
# print sentence_list
def sparse_feature_generation(length, dictionary):
    sparse = np.array([])
    feature_names = ['M1_ADJ', 'WBNULL', 'M1', 'M2', 'WBFL', 'WBF', 'WBL', 'WBO', 'BM1F', 'BM1L', 'AM2F', 'AM2L']
    sparse = np.append(sparse, dictionary['M1_ADJ'])
    if dictionary.has_key('WBNULL'):
        sparse = np.append(sparse, dictionary['WBNULL'])
    else:
        sparse = np.append(sparse, 0)
    for feature in feature_names[2:len(feature_names)]:
        feature_array = np.zeros(length)
        if dictionary.has_key(feature):
            feature_array[dictionary[feature]] = 1
        sparse = np.append(sparse, feature_array)
    sparse = csr_matrix(sparse)
    return sparse
a =   {
            "WBF": [
                1
            ],
            "BM1F": [
                4
            ],
            "M1": [
                2
            ],
            "M2": [
                3
            ],
            "BM1L": [
                2
            ],
            "WBO": [
                4,6
            ],
            "WBL": [
                2
            ],
            "M1_ADJ": [
                1
            ]


        }
matrix = sparse_feature_generation(10, a)
