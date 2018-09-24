# -*- coding: utf-8 -*-
import os
import codecs
from sentiment_model import sentimentModel
import data_utils
from gensim.models.word2vec import Word2Vec
import gensim
import numpy as np
from config import config

def train():
    print('loading word2vec')
    w2v_map, w2v_list = data_utils.load_w2v(config.data_output_path+'big_w2v.txt', config.dim_word)
    #print('loading vocabs')
    #vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label = data_utils.create_vocablulary(config.vocab_size, config.data_output_path)
    print('loading data')
    train_data, dev_data, _ = data_utils.load_data(config.data_output_path, w2v_map)
    print('--------------------')
    print(train_data[0])
    print('--------------------')
    print('building model')
    model = sentimentModel(config, w2v_list)
    model.build()
    print('Training...')
    model.train(train_data, dev_data)

def test():
    pass
    # test_data = data_utils.load_data(config.test_data)
    # model = sentimentModel(config)
    # model.build()
    # model.evaluate(test_data)

def data_preprocess():
    word_vocab = data_utils.trans_source_data(config.train_source_path, config.valid_source_path, config.test_source_path, config.data_output_path)
    print('Vocab size:', len(word_vocab))
    data_utils.filter_word2vec(config.data_output_path+'sgns.merge.word', config.data_output_path+'big_w2v.txt', word_vocab) 
    data_utils.learn_word2vec(config.data_output_path, config.data_output_path, config.dim_word)
    

if __name__ == '__main__':
    # data_preprocess()
    train()
   # annotate()

