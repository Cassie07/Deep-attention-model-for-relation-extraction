import os
import json
import numpy as np
import re

def NN_data_generation(product, relation, output_path):
    """

    :param product: including five different products' names
    :param relation: can be either positive or negative
    :param output_path: the output data set fot NN
    :return:
    """
    #location1, location2 are two keys in feature dictionary which represent two locations of mentions paiss in a sentence
    #sentence is the mention sentence
    append_feature = ['location1', 'location2', 'sentence']
    NN_data_set = np.array([])
    for json_file in sorted(os.listdir(os.path.abspath('..')+'/data/feature_'+relation+'/'+product)):
        with open(os.path.abspath('..')+'/data/feature_'+relation+'/'+product+'/'+json_file, 'r') as jf:
            feature_dict = json.load(jf)
            for id, feature_content in feature_dict.items():
               #id is the sentence's id
               #feature_content is the feature depend on each sentence
                id = id.encode('utf-8')
                for key, value in feature_content.items():
                    # key can be location, sentence, and a location tuple
                    # value is the response to the tuple
                    if key not in append_feature:
                        key = eval(key.encode('utf-8'))
                        #add_notation_sentence is the sentence contains the notation <e1>, </e1>, <e2>, </e2>
                        #exclude all the punctuations
                        add_notation_sentence = re.findall(r'\w+\-?\w+\-?\w+|\'?\w+' ,(feature_content['sentence']).encode('utf-8'))
                        NN_data_set = np.append(NN_data_set, (id+'.'+str(key)+'.'+product.split(' ')[0]))
                        NN_data_set = np.append(NN_data_set, '\t')
                        #when the word,which belongs to pairs, shows first in the sentence, add <e1> and </e1>to its sides.
                        add_notation_sentence[min(key)] = '<e1>'+add_notation_sentence[min(key)]+'</e1>'
                        #add <e2> and </e2> to the word's sides, which shows later
                        add_notation_sentence[max(key)] = '<e2>'+add_notation_sentence[max(key)]+'</e2>'
                        NN_data_set = np.append(NN_data_set, (' ').join(add_notation_sentence))
                        NN_data_set = np.append(NN_data_set, '\n')
                        if relation=='positive':
                            NN_data_set = np.append(NN_data_set, '+')
                        else:
                            NN_data_set = np.append(NN_data_set, '-')
                        if value['M1_ADJ']==[1]:
                            NN_data_set[-1] = NN_data_set[-1]+'(e1, e2)'+'\n'
                        else:
                            NN_data_set[-1] = NN_data_set[-1]+'(e2, e1)'+'\n'
    output = output_path+'/'+relation
    if os.path.exists(output):
        print 'dir: '+output+' exists'
    else:
        os.makedirs(output)
    with open(output+'/'+product+'.txt', 'w') as od:
        for item in NN_data_set:
            od.write(item)


if __name__ == '__main__':
    positive_feature_path = os.path.abspath('..')+'/data/feature_positive'
    negative_feature_path = os.path.abspath('..')+'/data'+'/feature_negative'
    output_path = os.path.abspath('..')+'/data'+'/dataset_NN'

    for product in sorted(os.listdir(positive_feature_path)):
        NN_data_generation(product, 'positive', output_path)
        NN_data_generation(product, 'negative', output_path)


