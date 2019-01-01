# this code is written for updating the dataset after labeling.
# some positive pairs may contain negative mentions
# Hence the lable of some mentions will be update
# And this code generates new dataset
import json
import os, sys
import re
import tensorflow as tf
import dataset_function

# distribute correct label for all postive mentions
def pos_neg_ids(path):
    pro_name=dataset_function.read_file(path)
    neg_train=[]
    pos_train=[]
    neg_test=[]
    pos_test=[]
    for i in pro_name:
        path2=path+'/'+i+'/'+'positive'+'.'+'json'
        json_data=open(path2).read()
        data = json.loads(json_data)
        #pairs=[i for i in data.keys()]
        #print(len(data.keys()))
        pair=[]
        path3='/projects/blstm/new_dataset/train_dataset/positive'+'/'+i+'.'+'txt'
        path4='/projects/blstm/new_dataset/test_dataset/positive'+'/'+i+'.'+'txt'
        with open(path3) as fp:
            for line in fp:
                pair.append(line.split( )[0])
                #pairs.remove(line.split( )[0])
        # train
        #print(len(pair))
        c=0
        for j in pair:
            #print(i)
            id_label=data[j]
            #print(id_label)
            for k in id_label:
                #print(k[1][0])
                if k[1][0]==1:
                    pos_train.append(k[0])
                else:
                    #print('it is 0')
                    neg_train.append(k[0])
                    #c=c+1
                    #print(c)
                    #print(k[0])
        # test
        pair=[]
        c=0
        with open(path4) as fp:
            for line in fp:
                pair.append(line.split( )[0])
                #pairs.remove(line.split( )[0])
        #print(len(pair))
        #print('*')
        #print(len(pair))
        for j in pair:
            id_label=data[j]
            for k in id_label:
                #print(k[1][0])
                #print(j[1][0])
                if k[1][0]==1:
                    pos_test.append(k[0])
                else:
                    #print("it is 0")
                    neg_test.append(k[0])
                    #c=c+1
                    #print(c)
        #print(pairs)
        #print('total_pos:',len(pos_train)+len(pos_test))
        #print('total_neg:',len(neg_train)+len(neg_test))
    return neg_train,pos_train,neg_test,pos_test # return mentions ids

def neg_ids(path):
    pro_name=dataset_function.read_file(path)
    neg_train=[]
    neg_test=[]
    for i in pro_name:
        pair=[]
        path_pair='/projects/blstm/Jiaxin_Liu/data/mention_id/pair_mid'+'/'+i+'/'+'negative'+'.'+'json'
        json_data=open(path_pair).read()
        pair_id = json.loads(json_data)
        path_neg_train='/projects/blstm/new_dataset/train_dataset/negative'+'/'+i+'.'+'txt'
        path_neg_test='/projects/blstm/new_dataset/test_dataset/negative'+'/'+i+'.'+'txt'
        # train
        with open(path_neg_train) as fp:
            for line in fp:
                pair.append(line.split( )[0])
        for i in pair:
            neg_train.append(pair_id[i][0])
        # test
        pair=[]
        with open(path_neg_test) as fp:
            for line in fp:
                pair.append(line.split( )[0])
        for i in pair:
            neg_test.append(pair_id[i][0])
        #print('neg_train:',len(neg_train))
        #print('neg_test:',len(neg_test))
    return neg_train,neg_test # return mentions ids

def combine(neg_train,neg_test,real_neg_train,real_neg_test):
    for i in neg_train:
        if i not in real_neg_train:
            real_neg_train.append(i)
    for j in neg_test:
        if j not in real_neg_test:
            real_neg_test.append(j)
    return real_neg_train,real_neg_test

# all pair of ids and mentions{id1:mention}
def generate_id_mention(path):
    pro_name=dataset_function.read_file(path)
    id_mention_ori={}
    for i in pro_name:
        id_mention_pos='/projects/blstm/Jiaxin_Liu/data/mention_id/mid_sentence/'+i+'/'+'positive'+'.'+'json'
        id_mention_neg='/projects/blstm/Jiaxin_Liu/data/mention_id/mid_sentence/'+i+'/'+'negative'+'.'+'json'
        json_data=open(id_mention_pos).read()
        data_pos = json.loads(json_data)
        #print(data_pos['b3d6250f8c0e2d723e032191144e6c20.(3, 15).Apex'])
        json_data=open(id_mention_neg).read()
        data_neg = json.loads(json_data)
        for j in data_pos.keys():
            id_mention_ori[j]=data_pos[j]
        for j in data_neg.keys():
            id_mention_ori[j]=data_neg[j]
    return id_mention_ori

#path='/projects/blstm/new_label'
#neg_train,pos_train,neg_test,pos_test=pos_neg_ids(path)
#print(len(neg_train))
#print(len(neg_test))
#print(len(pos_train))
#print(len(pos_test))
#print('*')
#real_neg_train,real_neg_test=neg_ids(path)
#print(len(real_neg_train))
#print(len(real_neg_test))
#print('*')
#neg_train_id,neg_test_id=combine(neg_train,neg_test,real_neg_train,real_neg_test)
#print(len(neg_train_id))
#print(len(neg_test_id))
#print(len(pos_train))
#print(len(pos_test))
#id_mention_ori=generate_id_mention(path)
