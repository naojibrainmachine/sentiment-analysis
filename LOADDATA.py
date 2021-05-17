import os
import fool
import math
import pandas as pd
import numpy as np
import random
import tensorflow as tf
def get_corpus_indices(data,chars_to_idx):
    """
    转化成词库索引
    
    """
    
    corpus_indices=[]
    for d in data:
        d=d.replace('\n','').replace('\r','').replace(' ','').replace('\u3000','')
        corpus_chars=fool.cut(d)
        corpus_chars=corpus_chars[0]
        corpus_indices.append([chars_to_idx[char] for char in corpus_chars])#语料索引，既读入的文本，并通过chars_to_idx转化成索引
    
    return corpus_indices

def data_format(data,labels):
    '''
    数据格式化，把整个批次的数据转化成最大数据长度的数据相同的数据长度（以-1进行填充）
    '''
    
    max_size=0
    new_data=[]
    
    #获取最大数据长度
    for x in data:
        if(max_size<len(x)):
            max_size=len(x)

    #格式化数据
    for x_t in data:
        if(abs(len(x_t)-max_size)!=0):
            for i in range(abs(len(x_t)-max_size)):
                x_t.extend([-1])
        new_data.append(tf.reshape(x_t,[1,-1]))

    new_labels = []

    #格式化标签
    for label in labels:
        new_labels.append(tf.reshape(label,[1,-1]))
    
    return new_data,new_labels

def get_data(data,labels,chars_to_idx,label_chars_to_idx,batch_size):
    '''
    一个批次一个批次的yield数据
    data:需要批次化的一组数据
    labels:data对应的情感类型
    chars_to_idx;词汇到索引的映射
    label_chars_to_idx;标签到索引的映射
    batch_size;批次大小
    '''
    num_example=math.ceil(len(data)/batch_size)
    
    example_indices=list(range(num_example))
    random.shuffle(example_indices)
    for i in example_indices:
        start=i*batch_size
        if start >(len(data)-1):
            start=(len(data)-1)
            
        
        end=i*batch_size+batch_size
        if end >(len(data)-1):
            end=(len(data)-1)+1
        
        X=data[start:end]
        Y=labels[start:end]
        
        X=get_corpus_indices(X,chars_to_idx)
        Y=get_corpus_indices(Y,label_chars_to_idx)
       
        yield X,Y #只是索引化的文本，且长度不一

def build_vocab(path):
    """
    构建词库
    path：数据集路径
    """
    df = pd.read_csv(path)

    #打乱索引
    rand=np.random.permutation(len(df))
    
    #获取数据总条数
    num_sum=len(df['label'])

    #获取所有数据，为构建词库做准备
    vocab = list(df['evaluation'])

    #获取所有标签
    labels=list(df['label'].unique())

    #获取训练数据，所有数据的90%为训练数据
    train_labels, train_vocab = list(df['label'].iloc[rand])[0:int(num_sum*0.9)], list(df['evaluation'].iloc[rand])[0:int(num_sum*0.9)]

    #获取测试数据，所有数据的10%为测试数据
    test_labels,test_vovab=list(df['label'].iloc[rand])[int(num_sum*0.9):num_sum], list(df['evaluation'].iloc[rand])[int(num_sum*0.9):num_sum]
    
    idx_to_chars=[]#索引到词汇的映射
    chars_to_idx={}#词汇到索引的映射

    #构建词库，用foolnltk进行分词
    for i in range(len(vocab)):
        corpus=vocab[i].replace('\n','').replace('\r','').replace(' ','').replace('\u3000','')
        corpus_chars=fool.cut(corpus)
        corpus_chars=corpus_chars[0]
        idx_to_chars.extend(corpus_chars)
    
        
    idx_to_chars=list(set(idx_to_chars))#索引到词汇的映射
    
    chars_to_idx=dict([(char,i) for i,char in enumerate(idx_to_chars)])#词汇到索引的映射

    label_idx_to_chars=list(set(labels))#索引到标签的映射
    
    label_chars_to_idx=dict([(char,i) for i,char in enumerate(label_idx_to_chars)])#标签到索引的映射
    
    vocab_size=len(idx_to_chars)#词库大小

    label_size=len(label_idx_to_chars)
    
    vocab.clear()
    return train_vocab,train_labels,test_labels,test_vovab,chars_to_idx,idx_to_chars,vocab_size,label_idx_to_chars,label_chars_to_idx,label_size

#build_vocab('data//data_single.csv')
#vocabulary,labels ,chars_to_idx,idx_to_chars,vocab_size,label_idx_to_chars,label_chars_to_idx,label_size=build_vocab('data//data_single.csv')
#get_data(data=vocabulary,labels=labels,chars_to_idx=chars_to_idx,label_chars_to_idx=label_chars_to_idx,batch_size=3)
