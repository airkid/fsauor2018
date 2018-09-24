#!/usr/bin/env python
# -*- coding: utf-8 -*-
class config():
    dim_word = 300
    dim_rnn = 150
    nlabels = 20
    dim_per_label = 4
    learning_rate = 0.00001
    lr_decay = 0.9
    dropout = 0.5
    batch_size = 16
    nepoch = 10

    filewriter_path = './graph/'
    output_path = './model/'
    data_output_path = './data/'
    vocab_size = 200000
    train_source_path = '/home/yanhr/contest/ai_challenger/sentiment/source_data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv'
    valid_source_path = '/home/yanhr/contest/ai_challenger/sentiment/source_data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv'
    test_source_path = '/home/yanhr/contest/ai_challenger/sentiment/source_data/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv'
