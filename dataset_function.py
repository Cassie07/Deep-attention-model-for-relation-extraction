import json
import os, sys
import re
import tensorflow as tf
#tf.flags.DEFINE_string("pair_mid_dir", "/projects/blstm/Jiaxin_Liu/data/mention_id/pair_mid", "Path of json file:{'pair':['mention_ids']}")
#tf.flags.DEFINE_string("id_mention_dir", "/projects/blstm/Jiaxin_Liu/data/mention_id/mid_sentence", "Path of json file: {'id':'mention'}")
#FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
#print("\nParameters:")
#for attr, value in sorted(FLAGS.__flags.items()): # sorted here is to sort the elements by their first letter
#	print("{} = {}".format(attr.upper(), value))
#print("")

# read all file names in a folder
def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	string = re.sub(r"[^A-Za-z0-9()<>/,!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()

def read_file(path):
    file_name=[]
    dirs=os.listdir(path)
    for file in dirs:
        if file=='.DS_Store'or file=='.ipynb_checkpoints'or file=='Untitled.ipynb':
            continue
        file_name.append(file)
    return file_name

# read all content in all files of a folder
def read_folder(path):
    name=read_file(path)
    list=[]
    dict={}
    for i in name:
        path1=path+'/'+i
        names=read_file(path1)
        #list.append(names)
        dict2={}
        for j in names:
            path2=path1+'/'+j
            json_data=open(path2).read()
            data = json.loads(json_data)
            dict2[j]=data
        dict[i]=dict2
    return dict

def generate_dataset(id_mention_dir,pair_mid_dir):
    # mid_sentence
    # generate two dict which contain ids with its mention
    #path='/projects/blstm/Jiaxin_Liu/data/mention_id/mid_sentence'
    id_mentions=read_folder(id_mention_dir)
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

    # pair_label
    #path='/projects/blstm/Jiaxin_Liu/data/mention_id/pair_label'
    #pair_label=read_folder(path)

    # read all mention ids into a list
    pair_mid=read_folder(pair_mid_dir)
    products=[i for i in pair_mid.keys()] # name of products
    #print(products)
    neg_id=[]
    pos_id=[]
    for i in products:
        label=[j for j in pair_mid[i].keys()]  # ['neg.json','pos.json']
        # neg
        neg_pair_id=pair_mid[i][label[0]] # ['id':'['id1',id2]',....]
        for k in neg_pair_id.values():
            for w in k:
                neg_id.append(w) # ['id1','id2']
        # pos
        pos_pair_id=pair_mid[i][label[1]]
        for k in pos_pair_id.values():
            for w in k:
                pos_id.append(w)
    #print(len(pos_id))
    #print(len(neg_id))
    # [['a7d8e305d570d95f2e66cc869353dc9a.(15, 17).Canon', 'he larger lens of the g3 gives better picture quality in low light and the <e1> 4 times </e11> optical <e2> zoom </e22> gets you just that much close', '+(e1, e2)']]
    # generate two list contain all info we need
    # [['id','mention','relation'],['id','mention','relation']]
    # neg
    dataset=[]
    for i in neg_id:
        pos=re.findall(r'[(](.*?)[)]', i) # position of two words
        pos=pos[0].split(',')
        mention=neg_id_mention[i]
        men_nopunc=re.findall(r'\w+\-?\w+\-?\w+|\'?\w+',mention) # a list contain all words in mentions(omit punctuations)
        p1=int(pos[0])
        p2=int(pos[1])
        if p1>p2:   # (14,3)
            # define relation
            relation='-(e2, e1)'
            # add position info to target word
            men_nopunc[p2]='<e1>'+men_nopunc[p2]+'</e1>'
            men_nopunc[p1]='<e2>'+men_nopunc[p1]+'</e2>'
            # combine all words in the list
            seperator = ' '
            sentence=seperator.join(men_nopunc)
        elif int(p1)==int(p1):   # (14,14)
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
        sentence = sentence.replace("<e1>", "<e1> ").replace("</e1>", " </e11>") # replace the front by the back
        sentence = sentence.replace("<e2>", "<e2> ").replace("</e2>", " </e22>")
        sentence = clean_str(sentence) # delete
        data=[i,sentence,relation]
        dataset.append(data)
    # pos
    for i in pos_id:
        pos=re.findall(r'[(](.*?)[)]', i) # position of two words
        pos=pos[0].split(',')
        mention=pos_id_mention[i]
        men_nopunc=re.findall(r'\w+\-?\w+\-?\w+|\'?\w+',mention) # a list contain all words in mentions(omit punctuations)
        p1=int(pos[0])
        p2=int(pos[1])
        if p1>p2:   # (14,3)
            relation='+(e2, e1)'
            men_nopunc[p2]='<e1>'+men_nopunc[p2]+'</e1>'
            men_nopunc[p1]='<e2>'+men_nopunc[p1]+'</e2>'
            seperator = ' '
            sentence=seperator.join(men_nopunc)
        elif p1==p2:   # (14,14)
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
        #print(sentence)
        sentence = sentence.replace("<e1>", "<e1> ").replace("</e1>", " </e11>") # replace the front by the back
        sentence = sentence.replace("<e2>", "<e2> ").replace("</e2>", " </e22>")
        sentence = clean_str(sentence) # delete
        data=[i,sentence,relation]
        dataset.append(data)
    return dataset
#print(generate_dataset(FLAGS.id_mention_dir,FLAGS.pair_mid_dir))
