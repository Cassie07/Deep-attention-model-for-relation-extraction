import json
import os, sys

# read all file names in a folder
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

# mid_sentence
path='/projects/blstm/Jiaxin_Liu-multi-instance-multi-label-3cdd2738a25e 2/data/mention_id/mid_sentence'
id_mentions=read_folder(path)
#print(id_mentions)
'''dict={
         'product_name1':{
                           'negative.json':{
                                            'mention_id1':'mention1',
                                            'mention_id2':'mention2'
                                            },
                           'positive.json':'{
                                             'mention_id1':'mention1',
                                             'mention_id2':'mention2'
                                             }
                          }
         'product_name2':{
                           'negative.json':{
                                            'mention_id1':'mention1',
                                            'mention_id2':'mention2'
                                            },
                           'positive.json':'{
                                             'mention_id1':'mention1',
                                             'mention_id2':'mention2'
                                             }
                          }
        }'''

# pair_label
path='/projects/blstm/Jiaxin_Liu-multi-instance-multi-label-3cdd2738a25e 2/data/mention_id/pair_label'
pair_label=read_folder(path)
#print(pair_label)
'''dict={
         'product_name1':{
                           'negative.json':{
                                            'pair1':'N',
                                            'pair2':'N'
                                            },
                           'positive.json':'{
                                             'pair1':'P',
                                             'pair2':'P'
                                             }
                          }
         'product_name2':{
                           'negative.json':{
                                            'pair1':'N',
                                            'pair2':'N'
                                            },
                           'positive.json':'{
                                             'pair1':'P',
                                             'pair2':'P'
                                             }
                          }
        }'''
# pair_mid
path='/projects/blstm/Jiaxin_Liu-multi-instance-multi-label-3cdd2738a25e 2/data/mention_id/pair_mid'
pair_mid=read_folder(path)
#print(pair_mid)

'''dict={
         'product_name1':{
                           'negative.json':{
                                            'pair1':[
                                                'id1',
                                                'id2'
                                            ],
                                            'pair2':[
                                                'id1'
                                            ]
                                            },
                           'positive.json':'{
                                             'pair1':[
                                                 'id1'
                                              ],
                                             'pair2':[
                                                 'id1',
                                                 'id2'
                                              ]
                                             }
                          }
         'product_name2':{
                           'negative.json':{
                                            'pair1':[
                                                'id1',
                                                'id2'
                                            ],
                                            'pair2':[
                                                'id1'
                                            ]
                                            },
                           'positive.json':'{
                                             'pair1':[
                                                 'id1'
                                              ],
                                             'pair2':[
                                                 'id1',
                                                 'id2'
                                              ]
                                             }
                          }
        }'''
