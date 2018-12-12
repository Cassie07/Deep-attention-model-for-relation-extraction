
import os, re, nltk
from Matrix_generation import lemmatization_nouns
from nltk.tag import pos_tag
from itertools import chain
from Matrix_generation import vocabulary_matrix

def positive_dict_generation(pairs_path):
    '''
    func: generate the positive pairs direction
            keys: adj
            values: [n, n, n, n]
    :param pairs_path:
    :return positive_pairs_dict
    '''
    with open(os.path.abspath('..') + '/data' + '/pairs'+'/'+pairs_path,'r') as fpair:
        #pairs: adj->n;n;n
        positive_pairs_dict = {}
        for pairs in fpair:
            pairs = pairs.strip()
            def filter_empty(s):
                return s and s.strip()
            entities = filter(filter_empty, re.split(r'\-+\>+|\;+', pairs))
            if len(entities) >=2:
                positive_pairs_dict[entities[0]] = entities[1:len(entities)]
    return positive_pairs_dict

def all_nous_extraction(id_path):
    '''
    func: extract all nouns from each sentences in 5 product files
    :param id_path:
    :return: nouns_dict
            keys: id
            values: [n, n, n]
    '''
    with open(os.path.abspath('..') + '/data' + '/id'+'/'+id_path, 'r') as fid:
        nouns_dict = {}
        for sentence in fid:
            sentence_pos = pos_tag(nltk.word_tokenize(lemmatization_nouns(sentence.split('\n')[0].split(' ',1)[1])))
            nouns = [x[0].encode('utf-8') for x in sentence_pos if ((x[1]=='NN') and x[0]!='i' and str.isalnum(x[0].encode('utf-8')))]
            if len(nouns)!=0:
                nouns_dict[sentence.split(' ')[0]] = nouns
    return nouns_dict

def filter_nonsense_nouns(black_list_path, nouns_dict):
    '''
    func: exclude all nouns which show up in the black list
    :param black_list_path:
    :param nouns_dict:
    :return: filter_noun_dict
            key: id
            value: [n,n,n]
    '''
    with open(os.path.abspath('..') + '/data' + '/negative_pairs' + '/black_list'+'/' + black_list_path, 'r') as fbl:
        black_nouns= [x.split('\n')[0] for x in fbl]
        filter_noun_dict = {}
        #keys: id, values: nouns in one sentence, exclude the ones in black list
        for keys, values in nouns_dict.items():
            #delete_nouns is the words show up in the black list,
            filter_noun = [x for x in values if x not in black_nouns]
            # print filter_noun
            if len(filter_noun)!=0:
                filter_noun_dict[keys] = filter_noun
    return filter_noun_dict

def delete_duplicate_value(listA):
        # return list(set(listA))
        return sorted(set(listA), key=listA.index)

def negative_pairs_generation(filter_noun_dict, entities_collection, product_row, positive_dict, sentence, output_path):
    '''
    filter_noun_dict:{} key: id,
                        value:[n, n, n, n]
    entities_collection:[[adj, n],[adj, n],[adj, n],...,[adj, n]]
    product_row: [[row1, row2],[row5, row2, row3],[],...,[]]
                      sentence row number response to each pairs.
    positive_dict = {} key: adj
                       value: [n,n,n,n,n]
    sentence: [[id, content],
               [id, content],]
    '''
    negative_pairs = {}
    adj = ''
    adj_seq = []
    sentence_same_adj = {}
    # collection of id, of which instance has the same adj mention.
    # key: adj, value:[id1, id2, id3]
    for pairs, rows in zip(entities_collection, product_row):
        adj_seq.append(pairs[0])
        if adj == '':
            adj = pairs[0]
            #sentence[row][0] is to get the sentence's id.
            #filter_noun_dict[sentence[row][0]] is to get all nouns in sentence, which response to the id
            #list(chain(*[[a,d],[d,c],[v,s]])) is to squeeze the list
            sentence_same_adj[adj] = list(chain(*[filter_noun_dict[sentence[row][0]] for row in rows if filter_noun_dict.has_key(sentence[row][0])]))
        elif adj != pairs[0]:
            adj = pairs[0]
            sentence_same_adj[adj] = list(chain(*[filter_noun_dict[sentence[row][0]] for row in rows if filter_noun_dict.has_key(sentence[row][0])]))
        else:
            sentence_same_adj[adj].extend(list(chain(*(filter_noun_dict[sentence[row][0]] for row in rows if filter_noun_dict.has_key(sentence[row][0])))))
            sentence_same_adj[adj] = delete_duplicate_value(sentence_same_adj[adj])
    adj_seq = delete_duplicate_value(adj_seq)
    for adj in adj_seq:
        negative_pairs[adj] = [n for n in sentence_same_adj[adj] if n not in positive_dict[adj]]
    if os.path.exists(output_path):
        print 'negative pairs path exist'
    else:
        print 'create the negative pairs path'
        os.makedirs(output_path)
    with open(output_path+'/'+product+'_pairs.txt', 'w') as fnegp:
        for key, value in negative_pairs.items():
            if len(value) != 0:
                value = delete_duplicate_value(value)
                fnegp.write(key+ '->')
                for a in range(1,len(value)-1):
                    fnegp.write(value[a] + ';')
                if range(len(value)) >1:
                    fnegp.write(value[-1])
                else:
                    continue
                fnegp.write('\n')


if __name__=='__main__':
    pairs_path = os.path.abspath('..') + '/data' + '/pairs'
    id_path = os.path.abspath('..') + '/data' + '/id'
    black_list_path = os.path.abspath('..') + '/data'+ '/negative_pairs'+'/black_list'
    negative_pair_path = os.path.abspath('..') + '/data' + '/negative_pairs'+'/pairs'

    (vectorizer, entities_collection, all_product_row, sentence, product_name, length) = vocabulary_matrix(pairs_path, id_path)
#    print entities_collection,all_product_row
    for pair, id, black_list, product in zip(sorted(os.listdir(pairs_path)), sorted(os.listdir(id_path)), sorted(os.listdir(black_list_path)), sorted(product_name)):
        positive_dict = positive_dict_generation(pair)
        noun_dictionary = filter_nonsense_nouns(black_list, all_nous_extraction(id))
        negative_pairs_generation(noun_dictionary, entities_collection[product], all_product_row[product], positive_dict, sentence, negative_pair_path)