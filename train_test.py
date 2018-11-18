# %load train_test.py
# %load train_test.py
# Generate training and testing dataset
#[['id','mention','relation'],['id','mention','relation']]
import dataset_function
import json
import os, sys
import re
import tensorflow as tf

#tf.flags.DEFINE_string("pair_mid_dir", "/projects/blstm/Jiaxin_Liu/data/mention_id/pair_mid", "Path of json file:{'pair':['mention_ids']}")
#tf.flags.DEFINE_string("id_mention_dir", "/projects/blstm/Jiaxin_Liu/data/mention_id/mid_sentence", "Path of json file: {'id':'mention'}")
#tf.flags.DEFINE_string("train_dir", "/projects/blstm/new_dataset/train_dataset", "Path of txt file which contains training data")
#tf.flags.DEFINE_string("test_dir", "/projects/blstm/new_dataset/test_dataset", "Path of txt file which contains testing data")
#tf.flags.DEFINE_string("dir", "/projects/blstm/new_dataset", "path of new dataset(same as Jiaxin)")
#tf.flags.DEFINE_string("pair_id_dir", "/projects/blstm/Jiaxin_Liu/data/mention_id/pair_mid", "Path of json file:{'pair':['mention_ids']}")
#FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
#print("\nParameters:")
#for attr, value in sorted(FLAGS.__flags.items()): # sorted here is to sort the elements by their first letter
#	print("{} = {}".format(attr.upper(), value))
#print("")


# pair_mid
#pair_mid=dataset_function.read_folder(FLAGS.pair_id_dir)
#print(pair_mid)
#a=pair_mid['Apex AD2600 Progressive-scan DVD player']
#b=a['positive.json']
#print(b['good_dd5.1'])

def load_pair_name(dir,pair_id_dir):
    pair_mid=dataset_function.read_folder(pair_id_dir)
    list=dataset_function.read_file(dir) #['test','train]
    train_dataset=[]
    test_dataset=[]
    for i in list:
        if i=='train_dataset':
            path=dir+'/'+i
            label=dataset_function.read_file(path) #['neg','pos',...]
            for j in label:
                path2=path+'/'+j
                product=dataset_function.read_file(path2) #['p1','p2',...]
                for k in product:
                    text_file = open(path2+'/'+k, "r")
                    lines = text_file.readlines()  # list of pairs: ['pair1/n','pair2/n',...]
                    train_dataset_pair=[]
                    for l in lines:  # omit '/n'
                        train_dataset_pair.append(l.split('\n')[0])
                    #print(train_dataset_pair)
                    for m in train_dataset_pair:  # match pair and find all mention_ids for pairs
                        dic_product=pair_mid[k.split('.')[0]] # key=product
                        dic_label=dic_product[j+'.json'] # key= label
                        for n in dic_label[m]:
                            train_dataset.append(n) # list of mention ids
            #print(len(train_dataset))
        else:
            path=dir+'/'+i
            label=dataset_function.read_file(path) #['neg','pos',...]
            for j in label:
                #print(j)
                path2=path+'/'+j
                product=dataset_function.read_file(path2) #['p1','p2',...]
                for k in product:
                    #print(k)
                    text_file = open(path2+'/'+k, "r")
                    lines = text_file.readlines()
                    test_dataset_pair=[]
                    for l in lines:
                        test_dataset_pair.append(l.split('\n')[0])
                    for m in test_dataset_pair:
                        #print(m)
                        dic_product=pair_mid[k.split('.')[0]] # key=product
                        dic_label=dic_product[j+'.json'] # key= label
                        for n in dic_label[m]:
                            test_dataset.append(n)
            #print(len(test_dataset))
            #print(len(test_dataset_pair))
    return train_dataset,test_dataset

def generate_dataset(train_dataset,test_dataset, id_mention_dir):
    # mid_sentence
    # generate two dict which contain ids with its mention
    #path='/projects/blstm/Jiaxin_Liu/data/mention_id/mid_sentence'
    id_mentions=dataset_function.read_folder(id_mention_dir)
    products=[i for i in id_mentions.keys()]
    neg_id_mention={}
    pos_id_mention={}
    id_mention={} # a dictionary contain all id:mention
    for i in products:
        label=[j for j in id_mentions[i].keys()]  # ['neg.json','pos.json']
        # neg
        neg_pair_id=id_mentions[i][label[0]] # ['id':'mention',....]
        se=[i for i in neg_pair_id.values()]
        for k in neg_pair_id.keys():
            neg_id_mention[k]=neg_pair_id[k]
            pos_pair_id=id_mentions[i][label[1]] # ['id':'mention',....]
        for k in pos_pair_id.keys():
            pos_id_mention[k]=pos_pair_id[k]

    train=[]
    test=[]
    # train
    for i in train_dataset:
        pos=re.findall(r'[(](.*?)[)]', i) # position of two words
        pos=pos[0].split(',')
        try:
            mention=neg_id_mention[i]
            sign=0
        except:
            mention=pos_id_mention[i]
            sign=1
        men_nopunc=re.findall(r'\w+\-?\w+\-?\w+|\'?\w+',mention) # a list contain all words in mentions(omit punctuations)
        p1=int(pos[0])
        p2=int(pos[1])
        if sign==0:
            if p1>p2:   # (14,3) 
                # define relation
                relation='-(e2, e1)'
                # add position info to target word
                men_nopunc[p2]='<e1>'+men_nopunc[p2]+'</e1>'
                men_nopunc[p1]='<e2>'+men_nopunc[p1]+'</e2>'
                # combine all words in the list
                seperator = ' '
                sentence=seperator.join(men_nopunc)
            elif int(p1)==int(p2):   # (14,14)
                relation='-(e2, e1)'
                men_nopunc[p2]='<e2><e1>'+men_nopunc[p2]+'</e1></e2>'
                seperator = ' '
                sentence=seperator.join(men_nopunc)
            else:
                relation='-(e1, e2)'  # (3,14)
                men_nopunc[p1]='<e1>'+men_nopunc[p1]+'</e1>'
                men_nopunc[p2]='<e2>'+men_nopunc[p2]+'</e2>'
                seperator = ' '
                sentence=seperator.join(men_nopunc)
        else:
            if p1>p2:   # (14,3) 
                # define relation
                relation='+(e2, e1)'
                # add position info to target word
                men_nopunc[p2]='<e1>'+men_nopunc[p2]+'</e1>'
                men_nopunc[p1]='<e2>'+men_nopunc[p1]+'</e2>'
                # combine all words in the list
                seperator = ' '
                sentence=seperator.join(men_nopunc)
            elif int(p1)==int(p2):   # (14,14)
                relation='+(e2, e1)'
                men_nopunc[p2]='<e2><e1>'+men_nopunc[p2]+'</e1></e2>'
                seperator = ' '
                sentence=seperator.join(men_nopunc)
            else:
                relation='+(e1, e2)'  # (3,14)
                men_nopunc[p1]='<e1>'+men_nopunc[p1]+'</e1>'
                men_nopunc[p2]='<e2>'+men_nopunc[p2]+'</e2>'
                seperator = ' '
                sentence=seperator.join(men_nopunc)
        sentence = sentence.replace("<e1>", "<e1> ").replace("</e1>", " </e11>") # replace the front by the back
        sentence = sentence.replace("<e2>", "<e2> ").replace("</e2>", " </e22>")
        sentence = dataset_function.clean_str(sentence) # delete
        data=[i,sentence,relation]
        train.append(data)
    # test
    for i in test_dataset:
        pos=re.findall(r'[(](.*?)[)]', i) # position of two words
        pos=pos[0].split(',')
        try:
            mention=neg_id_mention[i]
            sign=0
        except:
            mention=pos_id_mention[i]
            sign=1
        men_nopunc=re.findall(r'\w+\-?\w+\-?\w+|\'?\w+',mention) # a list contain all words in mentions(omit punctuations)
        p1=int(pos[0])
        p2=int(pos[1])
        if sign==0:
            if p1>p2:   # (14,3) 
                # define relation
                relation='-(e2, e1)'
                # add position info to target word
                men_nopunc[p2]='<e1>'+men_nopunc[p2]+'</e1>'
                men_nopunc[p1]='<e2>'+men_nopunc[p1]+'</e2>'
                # combine all words in the list
                seperator = ' '
                sentence=seperator.join(men_nopunc)
            elif int(p1)==int(p2):   # (14,14)
                relation='-(e2, e1)'
                men_nopunc[p2]='<e2><e1>'+men_nopunc[p2]+'</e1></e2>'
                seperator = ' '
                sentence=seperator.join(men_nopunc)
            else:
                relation='-(e1, e2)'  # (3,14)
                men_nopunc[p1]='<e1>'+men_nopunc[p1]+'</e1>'
                men_nopunc[p2]='<e2>'+men_nopunc[p2]+'</e2>'
                seperator = ' '
                sentence=seperator.join(men_nopunc)
        else:
            if p1>p2:   # (14,3) 
                # define relation
                relation='+(e2, e1)'
                # add position info to target word
                men_nopunc[p2]='<e1>'+men_nopunc[p2]+'</e1>'
                men_nopunc[p1]='<e2>'+men_nopunc[p1]+'</e2>'
                # combine all words in the list
                seperator = ' '
                sentence=seperator.join(men_nopunc)
            elif int(p1)==int(p2):   # (14,14)
                relation='+(e2, e1)'
                men_nopunc[p2]='<e2><e1>'+men_nopunc[p2]+'</e1></e2>'
                seperator = ' '
                sentence=seperator.join(men_nopunc)
            else:
                relation='+(e1, e2)'  # (3,14)
                men_nopunc[p1]='<e1>'+men_nopunc[p1]+'</e1>'
                men_nopunc[p2]='<e2>'+men_nopunc[p2]+'</e2>'
                seperator = ' '
                sentence=seperator.join(men_nopunc)
        sentence = sentence.replace("<e1>", "<e1> ").replace("</e1>", " </e11>") # replace the front by the back
        sentence = sentence.replace("<e2>", "<e2> ").replace("</e2>", " </e22>")
        sentence = dataset_function.clean_str(sentence) # delete
        data=[i,sentence,relation]
        test.append(data)
    return train,test

#train_d,test_d=load_pair_name(FLAGS.dir,FLAGS.pair_id_dir)
#id_mention_dir=FLAGS.id_mention_dir
#train,test=generate_dataset(train_d,test_d,id_mention_dir)
#print(len(train))
#print(test)