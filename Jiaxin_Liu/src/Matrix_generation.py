import os
import json
import codecs
import numpy as np
import re
import itertools
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def lemmatization_nouns(sents):
    sentence_lemma = []
    lemma = WordNetLemmatizer()
    sentence_tag = []
    a_list = 1
    if type(sents)!=list:
        a_list = 0
        sents = nltk.word_tokenize(sents)
    for word in sents:
        sentence_tag.extend(nltk.pos_tag([word]))
    for tag in sentence_tag:
        if tag[1]=='NNS':
            sentence_lemma.append(lemma.lemmatize(tag[0]))
        else:
            sentence_lemma.append(tag[0])
    if not a_list:
        sentence_lemma = ' '.join(sentence_lemma)
    return  sentence_lemma

def vocabulary_matrix(pairs_path, id_path):
    sentence = []
    sentence_number = {}
    # key: product name; value: [bag1, bag2, ..., bag_n]
    # each bag contains indices to sentences containing a particular pair
    all_product_row = {}
    entities_collection = OrderedDict()
    product_name = [x.split('_')[0] for x in sorted(os.listdir(pairs_path))]

    for pair_file, id_file in zip(sorted(os.listdir(pairs_path)), sorted(os.listdir(id_path))):
        sentence_num = 0
        entities_names = []

        with open(os.path.abspath(id_path)+'\\'+id_file,'r') as fid:
            for lines in fid:
                s_id = lines.split('\n')[0].split(' ',1)[0]
                s_content = lines.split('\n')[0].split(' ',1)[1]
                each_sentence = []
                sentence_num += 1
                each_sentence.append(s_id)
                each_sentence.append(lemmatization_nouns(s_content))
                sentence.append(each_sentence)
            sentence_number[id_file.split('_')[0]] = sentence_num
        with open(os.path.abspath(pairs_path)+'\\'+pair_file,'r') as fpair:
            for lines in fpair:
                # a line looks like adj->n;n;n
                lines = lines.strip('\n')
                def filter_empty(s):
                    return s and s.strip()
                entities = filter(filter_empty, re.split(r'\-+\>+|\;+', lines))
                entities = lemmatization_nouns(entities)
                for number in range(1, len(entities)):
                    # entities[0] is adj entites[number] are ns
                    entities_names.append([entities[0], entities[number]])

        entities_collection[pair_file.split('_')[0]] = entities_names

    corpus = np.array(sentence)[:, 1]

    vectorizer = CountVectorizer(token_pattern=r'\w+\-?\w+\-?\w+|\'?\w+', min_df=1)
    x = vectorizer.fit_transform(corpus)
    # row: represent all sentences from 5 product. col: all words in 5 products
    Vocabulary_Matrix = x.toarray()

    # offset into sentences of each product
    # a is not meaningful: a -> offset_product
    a = 0
    for product, each_product_entity in entities_collection.items():
        each_product_row = []
        for each_entity in np.array(each_product_entity):
            entity1_row = np.nonzero(Vocabulary_Matrix[a:a+sentence_number[product], vectorizer.vocabulary_.get(each_entity[0])])
            entity2_row = np.nonzero(Vocabulary_Matrix[a:a+sentence_number[product], vectorizer.vocabulary_.get(each_entity[1])])
            relative_sentence_index = set(entity1_row[0])&set(entity2_row[0])
            sentence_index = [x+a for x in relative_sentence_index]
            each_product_row.append(list(sentence_index))
        a = a+ sentence_number[product]
        all_product_row[product]= each_product_row
    return (vectorizer, entities_collection, all_product_row, sentence, product_name, len(Vocabulary_Matrix[0]))

def feature_generation(entity1, entity2, vectorizer, sentence_list):
    """
        func: generate the features' location index in the sentence
        input: entity1: first word in a pair
               entity2: second word in a pair
               vectorizer
               sentence_list - lst consist of a sentence
        output: the features' indexes
        annotation: WM1 - bag-of-words in M1
                    WM2 - bag-of-words in M2(M1 always appears before M2 in a sentence)
                    WBNULL - when no word in between(1 means no word in between)
                    WBFL - the only word in between when only one word in between
                    WBF - first word in between when at least two words in between
                    WBL - last word in between when at least two words in between
                    WBO - other words in between except first and last words when at least three words in between
                    BM1F - first word before M1
                    BM1L - second word before M1
                    AM2F - first word after M2
                    AM2L - second word after M2
                    M1_ADJ - '1' means M1 is the adj word, '0' means M1 is the noun word
            all the values of these keys means the column in the vectorizer matrix, it can check the column number by using:
            vectorizer.vocabulary_.get()
            for example:
                print vectorizer.vocabulary_.get('camera')
        """
    content = {}
    between = []
    if entity1 < entity2:
        M1 = entity1
        M2 = entity2
        content.update({'M1_ADJ': [1]})
    else:
        M1 = entity2
        M2 = entity1
        content.update({'M1_ADJ': [0]})
    if M2 - M1 <= 1:
        content.update({'WBNULL': [1]})
    elif M2 - M1 == 2:
        content.update({'WBFL': [vectorizer.vocabulary_.get(sentence_list[M1 + 1])]})
    else:
        content.update({'WBF': [vectorizer.vocabulary_.get(sentence_list[M1 + 1])]})
        content.update({'WBL': [vectorizer.vocabulary_.get(sentence_list[M2 - 1])]})
        if M2 - M1 > 4:
            for item in range(M1 + 2, M2 - 2, 1):
                between.append(vectorizer.vocabulary_.get(sentence_list[item]))
        else:
            between.append(vectorizer.vocabulary_.get(sentence_list[M1 + 2]))
        content.update({'WBO': between})
    if M1 >= 1:
        content.update({'BM1F': [vectorizer.vocabulary_.get(sentence_list[M1 - 1])]})
        if M1 >= 2:
            content.update({'BM1L': [vectorizer.vocabulary_.get(sentence_list[M1 - 2])]})
    if M2 <= (len(sentence_list) - 2):
        content.update({'AM2F': [vectorizer.vocabulary_.get(sentence_list[M2 + 1])]})
        if M2 <= len(sentence_list) - 3:
            content.update({'AM2L': [vectorizer.vocabulary_.get(sentence_list[M2 + 2])]})
    content.update({'M1': [vectorizer.vocabulary_.get(sentence_list[M1])]})
    content.update({'M2': [vectorizer.vocabulary_.get(sentence_list[M2])]})
    return content

def json_file_generation(pairs_path, id_path, output_path):
    """
        func: generate the bag of entities, if the pair of entities appears more than once in a sentence,
              then the bag will contain all the possible cases.
        input: path of the entities file, path of the sentence id file, path of the output bag .json file
        """
    vectorizer, entities_collection, all_product_row, sentence, product_name,_ = vocabulary_matrix(pairs_path, id_path)
    for product in product_name:
        bag_path = output_path + '/'+ product
        if os.path.exists(bag_path):
            print'dir: ' + bag_path + 'exists'
            print'dir: ' + bag_path + 'exists'
        else:
#            print 'create dir: ' + bag_path
            os.makedirs(bag_path)

        def find_pairs(x,y):
            return [a for a in range(len(y)) if y[a] == x]

        for entities_mention_index, each_entities in zip(all_product_row[product], entities_collection[product]):
            bag = {}
            for each_mention in entities_mention_index:
                mention = {}
                sentence_list = re.findall(r'\w+\-?\w+\-?\w+|\'?\w+', sentence[each_mention][1])
                entity1_location = find_pairs(each_entities[0], sentence_list)
                entity2_location = find_pairs(each_entities[1], sentence_list)
                location_collection =[two_location for two_location in itertools.product(entity1_location, entity2_location)]
                for index in location_collection:
                    mention['sentence'] = str(sentence[each_mention][1])
                    mention['location1'] = entity1_location
                    mention['location2'] = entity2_location
                    mention[str(tuple(index))] = feature_generation(index[0], index[1], vectorizer, sentence_list)
                bag[sentence[each_mention][0]] = mention

            with codecs.open(bag_path + '/' + each_entities[0]+'_'+ each_entities[1]+'.json', 'w') as outf:
                json.dump(bag, outf, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    pairs_path = os.path.abspath('..')+'/data'+'/pairs'
    negative_path = os.path.abspath('..')+'/data'+'/negative_pairs'+'/pairs'
    id_path = os.path.abspath('..')+'/data'+'/id'

    output_path_positive = os.path.abspath('..')+'/data'+'/feature'+'_positive'
    output_path_negative = os.path.abspath('..')+'/data'+'/feature'+'_negative'

    json_file_generation(pairs_path, id_path,output_path_positive)
    json_file_generation(negative_path, id_path, output_path_negative)