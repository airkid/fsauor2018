# -*- coding: utf-8 -*-
'''
模型
'''
import os
import numpy as np
import tensorflow as tf
from data_utils import minibatches
import logging
from config import config
from sklearn.metrics import f1_score

class sentimentModel(object):
    def __init__(self, config, embedding_data):
        self.config = config
        self.embedding_data = embedding_data
        # self.logger = logger
        # if logger is None:
        #     logger = logging.getlogger('logger')
        #     logger.setLevel(logging.DEBUG)
        #     logging.basicConfig(format='%(message)s', lebel=logging.DEBUG)

    def add_placeholders(self):
        self.data_x = tf.placeholder(tf.int32, shape=[None, None, None])
        self.sentence_length = tf.placeholder(tf.int32, shape=[None])
        self.max_sentence_length = tf.placeholder(tf.int32, shape=[])
        self.word_length = tf.placeholder(tf.int32, shape=[None])
        self.max_word_length = tf.placeholder(tf.int32, shape=[])
        self.data_y = tf.placeholder(tf.int32, shape=[None, self.config.nlabels, self.config.dim_per_label])
        self.dropout = tf.placeholder(tf.float32, shape=[])
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

    def add_word_embeddings_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embedding_data, name="_word_embeddings", dtype=tf.float32, trainable=True)
            self.word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.data_x, name="word_embeddings")
            #self.word_embeddings = tf.nn.dropout(self.word_embeddings, self.config.dropout)

    def compute_word_attention(self, hidden_list, attention_method='average'):
        if attention_method == 'average':
            return tf.reduce_mean(hidden_list, axis=1)
        elif attention_method == 'attention':
            with tf.variable_scope("attention_word"):
                hidden_state = [0 for i in range(self.config.nlabels)]
                for label in range(self.config.nlabels):
                    _hidden_list = tf.reshape(hidden_list, (-1,self.config.dim_rnn*2))
                    e_word = tf.reshape(tf.matmul(_hidden_list, tf.reshape(self.word_attentions[label],(300,1))), (-1, self.max_word_length))
                    a_word = tf.expand_dims(tf.nn.softmax(e_word, axis=-1),-1)
                    hidden_state[label] = tf.reduce_sum(hidden_list*a_word, axis=1)
                    #hidden_state[label] = tf.nn.dropout(hidden_state[label], self.config.dropout)
                return hidden_state


    def compute_sentence_attention(self, label, hidden_list, attention_method='average'):
        if attention_method == 'average':
            return tf.reduce_mean(hidden_list, axis=1)
        elif attention_method == 'attention':
            with tf.variable_scope("attention_sentence_"+str(label)):
                e_sentence = tf.matmul(hidden_list, self.sentence_attentions[label])
                a_sentence = tf.nn.softmax(e_sentence, axis=-1)
                hidden_state = tf.reduce_sum(hidden_list*a_sentence, axis=1)
                #hidden_state[label] = tf.nn.dropout(hidden_state[label], self.config.dropout)
            return hidden_state

    def add_logits_op(self):
        with tf.variable_scope("bi-lstm-word2sentence"):
            self.word_attentions = [tf.get_variable("word_attntion_"+str(i), shape=[self.config.dim_rnn*2], dtype=tf.float32) for i in range(self.config.nlabels)]
            word_embedding_shaped = tf.reshape(self.word_embeddings, (-1, self.max_word_length, self.config.dim_word))
            print('shape word_embedding_shaped:',word_embedding_shaped.get_shape())
            lstm_cell = tf.contrib.rnn.LSTMCell(self.config.dim_rnn)
            (output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell, word_embedding_shaped, sequence_length=self.word_length, dtype=tf.float32)
            self.word_lstm_output = self.compute_word_attention(tf.concat((output_fw, output_bw), axis=-1), attention_method='attention')
        
        self.sentence_attentions = [tf.get_variable("sentence_attntion_"+str(i), shape=[self.config.dim_rnn*2], dtype=tf.float32) for i in range(self.config.nlabels)]
        self.W = [tf.get_variable("W_"+str(i), shape=[2*self.config.dim_rnn, self.config.dim_per_label], dtype=tf.float32) for i in range(self.config.nlabels)]
        self.b = [tf.get_variable("b_"+str(i), shape=[self.config.dim_per_label], dtype=tf.float32, initializer=tf.zeros_initializer()) for i in range(self.config.nlabels)]
        self.logits = [0 for i in range(self.config.nlabels)]

        for label in range(self.config.nlabels):
            with tf.variable_scope("bi-lstm-sentence2doc"+str(label)):
                sentence_embedding_shaped = tf.reshape(self.word_lstm_output[label], (-1, self.max_sentence_length, self.config.dim_rnn*2))
                # print('shape sentence_embedding_shaped:',sentence_embedding_shaped.get_shape())
                lstm_cell = tf.contrib.rnn.LSTMCell(self.config.dim_rnn)
                (output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell, sentence_embedding_shaped, sequence_length=self.sentence_length, dtype=tf.float32)
                sentence_lstm_output = self.compute_sentence_attention(label, tf.concat((output_fw, output_bw), axis=-1), attention_method='average')
                # print('shape sentence_lstm_output:',sentence_lstm_output.get_shape())

            with tf.variable_scope("proj"+str(label)):
                lstm_feat = tf.reshape(sentence_lstm_output, [-1, 2*self.config.dim_rnn])
                self.logits[label] = tf.matmul(lstm_feat, self.W[label]) + self.b[label]
                # print('shape logits_:'+str(label),self.logits[label].get_shape())
        self.final_logits = tf.transpose(tf.convert_to_tensor(self.logits), perm=[1,0,2])
        print('shape final_logits:',self.final_logits.get_shape())

    def add_multi_softmax_loss(self):
        with tf.variable_scope("softmax_cross_entropy"):
            self.softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.data_y, logits=self.final_logits, dim=-1)
            print('shape softmax_cross_entropy:',self.softmax_cross_entropy.get_shape())
        self.loss = tf.reduce_sum(tf.reduce_mean(self.softmax_cross_entropy, axis=0))
        tf.summary.scalar("loss", self.loss)

    def add_pred_op(self):
        self.soft_max_logits = tf.nn.softmax(self.final_logits, dim=-1)
        self.labels_pred = tf.argmax(self.soft_max_logits, axis=-1)
        print('shape labels_pred:',self.labels_pred.get_shape())
        _accuracy = tf.equal(self.labels_pred, tf.argmax(self.data_y, axis=-1))
        print('shape _accuracy:',_accuracy.get_shape())
        self.acc = tf.reduce_mean(tf.cast(_accuracy, tf.float32), axis=0)
        self.mean_acc = tf.reduce_mean(self.acc)
        print('shape acc:',self.acc.get_shape())
        tf.summary.scalar("mean_acc", self.mean_acc)

    def add_train_op(self):
        with tf.variable_scope("train_op"):
            opt = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            self.train_op = opt.minimize(self.loss)

    def add_init_op(self):
        self.init = tf.global_variables_initializer()

    def add_summary(self, sess):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.filewriter_path, sess.graph)

    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_multi_softmax_loss()
        self.add_pred_op()
        self.add_train_op()
        self.add_init_op()

    def get_feed_dict(self, batch_num, docs, labels=None, lr=None, dropout=None, one=False):
        print('Getting feed dict: '+str(batch_num))
        sentence_length = []
        word_length = []
        for i in range(len(docs)):
            sentence_length.append(len(docs[i]))
            word_length.append([len(sen) for sen in docs[i]])
        max_sentence_length = max(sentence_length)
        max_word_length = 0
        for i in word_length:
            for j in i:
                max_word_length = max(max_word_length, j)
        for i in range(len(docs)):
            for j in range(len(docs[i])):
                docs[i][j].extend([0 for k in range(max_word_length-len(docs[i][j]))])
            for j in range(max_sentence_length - len(docs[i])):
                docs[i].append([0 for k in range(max_word_length)])
            word_length[i].extend([0 for k in range(max_sentence_length - len(word_length[i]))])
        docs = np.array(docs)
        sentence_length = np.array(sentence_length)
        word_length = np.reshape(np.array(word_length), -1)
        feed = {
            self.data_x: docs,
            self.sentence_length: sentence_length,
            self.word_length: word_length,
            self.max_sentence_length: max_sentence_length,
            self.max_word_length: max_word_length,
        }
        if labels is not None:
            feed[self.data_y] = labels
        if lr is not None:
            feed[self.learning_rate] = lr
        if dropout is not None:
            feed[self.dropout] = dropout
        print('Done')
        return feed

    def predict_batch(self, batch_num, sess, words, labels):
        fd = self.get_feed_dict(batch_num, words, labels=labels, dropout=1.0)
        pred, acc = sess.run([self.labels_pred, self.acc], feed_dict=fd)
        return pred, acc

    def f1_update(self, pred, _labels, TP, FP, FN):
        labels = np.argmax(np.reshape(_labels,(-1, self.config.nlabels, self.config.dim_per_label)), axis=-1)
        for i in range(len(pred)):
            for j in range(self.config.nlabels):
                if int(pred[i][j]) == int(labels[i][j]):
                    TP[j][int(pred[i][j])] += 1
                else:
                    FP[j][int(pred[i][j])] += 1
                    FN[j][int(labels[i][j])] += 1
        return TP, FP, FN

    def compute_f1(self, TP, FP, FN):
        precision, recall = [np.zeros((self.config.nlabels,self.config.dim_per_label)) for i in range(2)]
        for i in range(self.config.nlabels):
            for j in range(self.config.dim_per_label):
                if TP[i][j]+FP[i][j] != 0:
                    precision[i][j] = float(TP[i][j])/(TP[i][j]+FP[i][j])
                else:
                    precision[i][j] = 0
                if TP[i][j]+FN[i][j] != 0:
                    recall[i][j] = float(TP[i][j])/(TP[i][j]+FN[i][j])
                else:
                    recall[i][j] = 0
        f1 = np.zeros(self.config.nlabels)
        for i in range(self.config.nlabels):
            for j in range(self.config.dim_per_label):
                if precision[i][j]+recall[i][j] != 0:
                    _f1 = 2*float(precision[i][j])*recall[i][j]/float(precision[i][j]+recall[i][j])
                else:
                    _f1 = 0
                f1[i] += _f1
            if f1[i] != 0:
                f1[i] /= self.config.dim_per_label 
        return f1

    def update_f1(self, pred, _labels, pred_list, label_list):
        labels = np.argmax(np.reshape(_labels,(-1, self.config.nlabels, self.config.dim_per_label)), axis=-1)
        for i in range(self.config.nlabels):
            pred_list[i].extend([_pred[i] for _pred in pred])
            label_list[i].extend([_label[i] for _label in labels])
        return pred_list, label_list

    def run_eval(self, sess, test):
        accs = []
        # TP, FP, FN = [np.zeros((self.config.nlabels,self.config.dim_per_label)) for i in range(3)]
        label_list = [[] for i in range(self.config.nlabels)]
        pred_list = [[] for i in range(self.config.nlabels)]
        correct_preds, total_correct, total_preds = 0,0,0
        for i, (words, labels) in enumerate(minibatches(test, self.config.batch_size)):
            pred,  _acc= self.predict_batch(i, sess, words, labels)
            accs.append(_acc)
            # TP, FP, FN = self.f1_update(pred, labels, TP, FP, FN)
            pred_list, label_list = self.update_f1(pred, labels, pred_list, label_list)
        acc = np.mean(np.array(accs), axis=0)
        # f1 = self.compute_f1(TP, FP, FN)
        # print(label_list, pred_list)
        f1 = [f1_score(label_list[i], pred_list[i], average='macro') for i in range(self.config.nlabels)]
        return acc, f1

    def run_epoch(self, sess, train, dev, epoch):
        nbatches = (len(train) + self.config.batch_size -1) / self.config.batch_size
        total_loss = 0.0
        batch_cnt = 0
        # acc, f1 = self.run_eval(sess, dev)
        # print('In valid data: ')
        # print('Accuracy: ',acc,'\n','Mean Accuracy: ', np.mean(acc))
        # print('F1 Score: ',f1,'\n','Macro F1 Score: ', np.mean(f1))
        for i, (words, labels) in enumerate(minibatches(train, self.config.batch_size)):
            fd = self.get_feed_dict(i, words, labels=labels, lr=self.config.learning_rate, dropout=self.config.dropout)
            _, loss, mean_acc, summary = sess.run([self.train_op, self.loss, self.mean_acc, self.merged], feed_dict=fd)
            total_loss += loss
            if i % 300 == 0 and i !=0:
                print('In valid data: ')
                acc, f1 = self.run_eval(sess,dev)
                print('Accuracy: ',acc,'\n','Mean Accuracy: ', np.mean(acc))
                print('F1 Score: ',f1,'\n','Macro F1 Score: ', np.mean(f1))
                #self.file_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=['eval_acc_label_'+str(i) for i in range(self.config.nlabels)],simple_value=acc)]),epoch)
            batch_cnt += 1
            if i % 20 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)
            if i % 100 == 0:
                print('epoch: %d, batch: %d, mean_acc: %.2f'%(epoch, i, mean_acc))
                print("batch {}, loss {:04.2f}.".format(i, float(total_loss)/batch_cnt))
        acc, f1 = self.run_eval(sess, dev)
        print('In valid data: ')
        print('Accuracy: ',acc,'\n','Mean Accuracy: ', np.mean(acc))
        print('F1 Score: ',f1,'\n','Macro F1 Score: ', np.mean(f1))
    #self.file_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='eval_acc',simple_value=acc)]),epoch)
        # print("- dev acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * 0))
        return acc

    def train(self, train, dev):
        best_score = 0
        saver = tf.train.Saver()
        early_stopping_round = -1
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.config.output_path)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt)
                saver.restore(sess, self.config.output_path)
            else:
                print('Begin to initialize ...')
                sess.run(self.init)
            self.add_summary(sess)
            for epoch in range(self.config.nepoch):
                print("Epoch {:} out of {:}".format(epoch + 1, self.config.nepoch))
                
                acc = self.run_epoch(sess, train, dev, epoch)
                
                if np.mean(acc) > best_score:
                    if not os.path.exists(self.config.output_path):
                        os.makedirs(self.config.output_path)
                    saver.save(sess, self.config.output_path)
                    best_score = np.mean(acc)
                    print("- new best score! ",acc)

                self.config.learning_rate *= self.config.lr_decay
    
    def evaluate(self, test):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print("Testing model over test set")
            saver.restore(sess, self.config.output_path)
            acc, f1 = self.run_eval(sess, test)
            print('Accuracy: ',acc,'\n','Mean Accuracy: ', np.mean(acc),'\n','F1 Score: ',f1,'\n','Macro F1 Score: ', np.mean(f1))
            #print("- test acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * 0))
