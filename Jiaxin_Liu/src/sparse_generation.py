import os
import json
import pickle
from Matrix_generation import vocabulary_matrix
import numpy as np
from scipy.sparse import coo_matrix, vstack
from scipy.sparse import csr_matrix

def dataset_separate(json_file, pairs_type, test_size = 0.5):
    """

    :param json_file: the input of the json files
    :param test_size: the proportion of test set
    :param pairs_type: can either separate positive or negative data set.
    :return: dataset is a list contains all pairs' names
             train_Set is a list contains all training pairs' names
    """
    if pairs_type=='positive':
        #dataset is a list contains all pairs names in each product
        dataset = os.listdir(os.path.abspath('..')+'/data/feature_'+pairs_type + '/'+json_file)
    else:
        dataset = os.listdir(os.path.abspath('..') + '/data/feature_'+pairs_type + '/' + json_file)
        #train_set is a list contain all pairs names which constitute as the training set
    train_set = np.random.choice(dataset, int(len(dataset)*test_size), replace=False)
    test_set = list(set(dataset).difference(set(train_set)))
    return dataset, train_set, test_set


def sparse_feature_generation(length, dictionary):
    """
    generate the sparse feature row for each mention
    :param length: the scale of the vocabulary matrix
    :param dictionary: the feature dictionary key: feature names, value: list of index
    :return: sparse row format: csr
    """
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

def sparse_matrix_generation(relation, i_json_file_path, product, o_dictionary_papth, length):
    """
    this function will generate three mention annotations
    :param relation: can be positive or negative
    :param i_json_file_path: is the input of json file
    :param product: can be five products' names
    :param o_dictionary_papth: the root path
    :param length: the scale of vocabulary matrix
    :return:
    """
    append_key = ['location1', 'location2', 'sentence']
    pair_mid = {}
    #key:mention_id, value: sparse feature vector
    mid_sparse ={}
    #key: pair, value: label
    pair_label = {}
    #key: mention_id, value: sentence
    mid_sentence = {}
    j_path = i_json_file_path+'_'+relation+'/'+product
    for pairs in sorted(os.listdir(j_path)):
        with open(j_path+'/'+pairs, 'r') as jf:
            bag = json.load(jf)
            pair_mid[pairs.split('.')[0]] = []
            for id, sentence in bag.items():
                for index, feature_value in sentence.items():
                    if index.encode() not in append_key:
                        mention_id = id+'.'+index.encode()+'.'+product.split(' ')[0]
                        pair_mid[pairs.split('.')[0]].append(mention_id)
                        mid_sparse[mention_id] = sparse_feature_generation(length, feature_value)
                        mid_sentence[mention_id] = sentence['sentence']
        if not pair_label.has_key(pairs.split('.')[0]):
            if relation == 'positive':
                pair_label[pairs.split('.')[0]] = 'p'
            else:
                pair_label[pairs.split('.')[0]] = 'N'
        else:
            print pairs+' duplicate'
    #save the pair_mid dictionary into mention_id/pair_mid/product1/positive.json
    pair_mid_path = o_dictionary_papth+'/pair_mid'+ '/' +product
    if not os.path.exists(pair_mid_path):
        print 'generate '+ pair_mid_path
        os.makedirs(pair_mid_path)
    with open(pair_mid_path+'/'+relation+'json', 'w') as p:
        json.dump(pair_mid, p, indent=4)
    #save the mid_sparse dictionary into mention_id/mid_sparse/product1/positive.pickle
    mid_sparse_path = o_dictionary_papth+'/mid_sparse'+'/'+ product
    if not os.path.exists(mid_sparse_path):
        print 'generate '+ mid_sparse_path
        os.makedirs(mid_sparse_path)
    with open(mid_sparse_path+'/'+relation+'.pickle', 'w') as m:
        pickle.dump(mid_sparse, m)
    #save the pair_label dictionary into mention_id/pair_label/product1/positive.json
    pair_label_path = o_dictionary_papth+'/pair_label'+'/'+product
    if not os.path.exists(pair_label_path):
        print 'generate '+ pair_label_path
        os.makedirs(pair_label_path)
    with open(pair_label_path+'/'+relation+'.json', 'w') as l:
        json.dump(pair_label, l, indent=4)
    mid_sentence_path = o_dictionary_papth+'/mid_sentence'+'/'+product
    if not os.path.exists(mid_sentence_path):
        print 'generate '+mid_sentence_path
        os.makedirs(mid_sentence_path)
    with open(mid_sentence_path+'/'+relation+'.json', 'w') as s:
        json.dump(mid_sentence, s, indent=4)

def test_pair_generation(test_set, relation, product, output_path):
    """

    :param test_set: a list contains names of all test pairs' for each product
    :param relation: can be 'positive' or 'negative'
    :param product: five products' names
    :param output_path: the path of the test data set
    :return:
    """
    output = output_path+'/'+relation
    if not os.path.exists(output):
        os.makedirs(output)
        print 'dir: '+output+' set'
    with open(output+'/'+product+'.txt', 'w') as tf:
        for test_pairs in test_set:
            tf.write(test_pairs.split('.json')[0]+'\n')

if __name__ == '__main__':
    pairs_path = os.path.abspath('..') + '/data'+'/pairs'
    id_path = os.path.abspath('..') + '/data'+'/id'
    json_path = os.path.abspath('..')+ '/data' + '/feature'

    positive_training_set = os.path.abspath('..')+'/data/'
    test_bag_path = os.path.abspath('..')+'/data/LR/test_dataset'
    train_pickle_path = os.path.abspath('..')+'/data/LR/train'
    output_dictionary = os.path.abspath('..')+'/data/mention_id'

#    _,_, _, _, _, length = vocabulary_matrix(pairs_path, id_path)
    sparse_feature_matrix = coo_matrix((1,3217*10+2))
    mention_label = np.array([0])

    for product in sorted(os.listdir(pairs_path)):
        sparse_matrix_generation('positive', json_path, product.split('_')[0], output_dictionary, length = 3217)
        sparse_matrix_generation('negative', json_path, product.split('_')[0], output_dictionary, length = 3217)
#        positive_dataset, positive_train_set, positive_test_set = dataset_separate(product, 'positive')
#        sparse_matrix_generation(positive_train_set, product, length=3217, pairs_type='positive')
#        test_pair_generation(positive_test_set, 'positive', product, test_bag_path)

#        negative_dataset, negative_train_set, negative_test_set = dataset_separate(product, 'negative')
#        sparse_matrix_generation(negative_train_set, product, length=3217, pairs_type='negative')
#        test_pair_generation(negative_test_set, 'negative', product, test_bag_path)

#    generate_pickle_file(sparse_feature_matrix.tocsr(), train_pickle_path, 'X')
#    generate_pickle_file(mention_label, train_pickle_path, 'y')
