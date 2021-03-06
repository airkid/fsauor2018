# -*- coding: utf-8 -*-
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8') #gb2312
import codecs
import random
import os
import numpy as np
import pickle
import pandas as pd
import numpy as np
from collections import Counter
from gensim.models.word2vec import Word2Vec
from config import config
from nltk.tokenize import sent_tokenize, word_tokenize


PAD_ID = 0
UNK_ID = 1
_PAD = "_PAD"
_UNK = "UNK"

def create_vocablulary(vocab_size, data_path, save_path='./vocab_cache', file_name='vocab.pkl'):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    file_save_path = save_path+'/'+file_name
    if os.path.exists(file_save_path):
        with open(file_save_path,'rb') as data_f:
            return pickle.load(data_f)
    else:
        vocabulary_word2index={}
        vocabulary_index2word={}
        vocabulary_word2index[_PAD]=PAD_ID
        vocabulary_index2word[PAD_ID]=_PAD
        vocabulary_word2index[_UNK]=UNK_ID
        vocabulary_index2word[UNK_ID]=_UNK

        vocabulary_label2index={}
        vocabulary_index2label={}

        #1.load raw data
        file_object = codecs.open(data_path+'train_tag.pkl', mode='rb')
        train_data = pickle.load(file_object)
        file_object.close()
        #2.loop each line,put to counter
        c_inputs=Counter()
        c_labels=Counter()
        for line in train_data:
            input_list = line[0].strip().split(" ")
            input_list = [x.strip().replace(" ", "") for x in input_list if x != '']
            label_list=line[1]
            c_inputs.update(input_list)
            c_labels.update(label_list)
        #return most frequency words
        vocab_list=c_inputs.most_common(vocab_size)
        label_list=c_labels.most_common()
        #put those words to dict
        for i,tuplee in enumerate(vocab_list):
            word,_=tuplee
            vocabulary_word2index[word]=i+2
            vocabulary_index2word[i+2]=word
        for i,tuplee in enumerate(label_list):
            label,_=tuplee;label=str(label)
            vocabulary_label2index[label]=i
            vocabulary_index2label[i]=label

        #save to file system if vocabulary of words not exists.
        if not os.path.exists(file_save_path):
            with open(file_save_path, 'ab') as data_f:
                pickle.dump((vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label), data_f)
    return vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label

def filter_word2vec(source_path, save_path, word_vocab):
    print("Filter word2vec")
    source_file = codecs.open(source_path, 'r', 'utf-8')
    save_file = codecs.open(save_path, 'w', 'utf-8')
    cnt = 0
    for line in source_file.readlines()[1:]:
        tmp = line.find(' ')
        word = line[:tmp]
        if word in word_vocab:
            cnt += 1
            save_file.write(line)
    source_file.close()
    save_file.close()
    print('Filter Num: ',cnt)

def learn_word2vec(train_data_path, save_path, dimword, min_count=3):
    print("Learning word2vec dimword = %d min_count = %d"%(dimword,min_count))
    file_object = codecs.open(train_data_path+'train_tag.pkl', mode='rb')
    train_data = pickle.load(file_object)
    file_object.close()
    random.shuffle(train_data)
    X = []
    for data_line in train_data:
        # line = [id,content,label......]
        content_list = data_line[0].strip().split(" ")
        X.append([word.strip() for word in content_list])
    w2v_model = Word2Vec(X, size=dimword, min_count=min_count)
    w2v_model.wv.save_word2vec_format(save_path+'w2v.txt', binary=False)

def load_w2v(path, dimword):
    file_object = codecs.open(path, 'r', 'utf-8')
    w2v_map = {'UNK':0}
    cnt = 0
    w2v_list = [[0.0 for i in range(dimword)]]
    for line in file_object.readlines()[1:]:
        cnt += 1
        tmp = line.find(' ')
        w2v_map[line[:tmp]] = cnt
        vec = line[tmp+1:-2].split(' ')
        vec = [float(num) for num in vec]
        w2v_list.append(vec)
    return w2v_map, w2v_list

def word2id_load(filepath, w2v_map):
    file_object = codecs.open(filepath, mode='rb')
    data = pickle.load(file_object)
    for index in range(len(data)):
        x = data[index][0].strip().split(' ')
        data[index][0] = [w2v_map[word] if word in w2v_map else w2v_map['UNK'] for word in x]
    file_object.close()
    return data

def load_data(data_path, vocabulary_word2index):
    train_data = word2id_load(data_path+'train_tag.pkl', vocabulary_word2index)
    valid_data = word2id_load(data_path+'valid_tag.pkl', vocabulary_word2index)
    test_data = word2id_load(data_path+'test_tag.pkl', vocabulary_word2index)
    return train_data, valid_data, test_data

def _trans(source_data, label_list, word_vocab, label=False, length_limit=-1):
    trans_data = []
    for index, row in source_data.iterrows():
        _content = row['content'].strip()
        _content = jieba.cut(_content)
        x = ' '.join(_content)
        word_vocab.update(x.split(' '))
        one_hot_label = np.zeros(4*len(label_list))
        if label:
            for index, label in enumerate(label_list):
                one_hot_label[(int(row[label])+2)+index*4] = 1.0
        if length_limit != -1 and len(x.split(' ')) <= length_limit:
            continue
        trans_data.append([x,one_hot_label])
    return trans_data, word_vocab

def trans_source_data(train_data_path, valid_data_path, test_data_path, save_path):
    print('Transforming source_data (tag and save)')
    word_vocab = set()
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    source_train = pd.read_csv(train_data_path)
    source_valid = pd.read_csv(valid_data_path)
    source_test = pd.read_csv(test_data_path)
    output_train = codecs.open(save_path+'train_tag.pkl', 'wb')
    output_valid = codecs.open(save_path+'valid_tag.pkl', 'wb')
    output_test = codecs.open(save_path+'test_tag.pkl', 'wb')

    label_list = source_train.columns.tolist()[2:]

    train, word_vocab = _trans(source_train, label_list, word_vocab, label=True, length_limit=200)
    valid, word_vocab = _trans(source_valid, label_list, word_vocab, label=True, length_limit=200)
    test, word_vocab = _trans(source_test, label_list, word_vocab, label=False, length_limit=200)

    print('Train data: ',len(train))
    print('Valid data: ',len(valid))
    print('Test data: ',len(test))

    pickle.dump(train, output_train)
    pickle.dump(valid, output_valid)
    pickle.dump(test, output_test)
    output_train.close()
    output_valid.close()
    output_test.close()
    return word_vocab

def minibatches(data, minibatch_size):
    x_batch, y_batch = [], []
    random.shuffle(data)
    for (x,y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch

if __name__ == '__main__':
    train_path = '/home/yanhr/contest/ai_challenger/sentiment/source_data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv'
    valid_path = '/home/yanhr/contest/ai_challenger/sentiment/source_data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv'
    test_path = '/home/yanhr/contest/ai_challenger/sentiment/source_data/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv'
    trans_source_data(train_path, valid_path, test_path, '/home/yanhr/contest/ai_challenger/fsauor2018/RNN/data')
