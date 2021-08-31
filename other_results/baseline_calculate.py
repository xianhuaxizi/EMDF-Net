# coding:utf8
#
# Run this code to get the final results reported in our ijcai paper.
from io import open
import string
import re
import random
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pprint, copy
from utils import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def get_event_chains(event_list):
    return ['%s_%s' % (ev[0], ev[2]) for ev in event_list]


def get_word_embedding(word, word_id, id_vec, emb_size):
    if word in word_id:
        return id_vec[word_id[word]]
    else:
        return np.zeros(emb_size, dtype=np.float32)


def get_vec_rep(questions, word_id, id_vec, emb_size, predict=False):
    rep = np.zeros((5 * len(questions), 9, emb_size), dtype=np.float32)
    correct_answers = []
    for i, q in enumerate(questions):
        context_chain = get_event_chains(q[0])
        choice_chain = get_event_chains(q[1])
        correct_answers.append(q[2])
        for j, context in enumerate(context_chain):
            context_vec = get_word_embedding(context, word_id, id_vec, emb_size)
            rep[5 * i:5 * (i + 1), j, :] = context_vec
        for k, choice in enumerate(choice_chain):
            choice_vec = get_word_embedding(choice, word_id, id_vec, emb_size)
            rep[5 * i + k, -1, :] = choice_vec
    if not predict:
        input_data = Variable(torch.from_numpy(rep))
    else:
        input_data = Variable(torch.from_numpy(rep), volatile=True)
    correct_answers = Variable(torch.from_numpy(np.array(correct_answers)))
    return input_data, correct_answers

def process_test(scores, test_index):
    for index in test_index:
        scores[index] = np.min(scores)
    return scores

def get_acc(scores, correct_answers, name='scores', save=False):
    selections = np.argmax(scores, axis=1)
    num_correct = int(np.sum(selections == correct_answers))
    if save:
        pickle.dump((selections == correct_answers), open('./scores/' + name, 'wb'), 2)
    samples = len(correct_answers)
    accuracy = float(num_correct) / samples * 100.
    # print ("%d / %d correct: %f" % (num_correct, samples, accuracy))
    print("%s Accuracy：%.2f" % (name, accuracy))

if __name__ == '__main__':
    data_path = '../data/'
    test_index = pickle.load(open(data_path+'test_index.pickle', 'rb'))
    test_data = Data_data(pickle.load(open(data_path+'vocab_index_test.data', 'rb')))
    data = test_data.all_data()
    correct_answers = data[2].cpu().data.numpy()

    home_path = './'
    scores1 = pickle.load(open(home_path+ 'event_comp_test.scores', 'rb'), encoding='bytes')
    scores1 = process_test(scores1, test_index)
    get_acc(scores1, correct_answers, 'event_comp')

    scores2 = pickle.load(open(home_path+ 'PairwiseLSTM_test.scores', 'rb'), encoding='bytes')
    scores2 = process_test(scores2, test_index)
    get_acc(scores2, correct_answers, 'PairwiseLSTM')

    scores3 = pickle.load(open(home_path+ 'SGNN_test.scores2', 'rb'), encoding='bytes')
    scores3 = process_test(scores3, test_index)
    get_acc(scores3, correct_answers, 'SGNN')

    scores4 = pickle.load(open(home_path+ 'MCer_test.scores2', 'rb'), encoding='bytes')
    scores4 = process_test(scores4, test_index)
    get_acc(scores4, correct_answers, 'MCer')

    scores6 = pickle.load(open(home_path+ 'EMDF-Net_test.scores4', 'rb'), encoding='bytes')
    scores6 = process_test(scores6, test_index)
    get_acc(scores6, correct_answers, 'EMDF-Net')

    scores1 = preprocessing.scale(scores1)
    scores2 = preprocessing.scale(scores2)
    scores3 = preprocessing.scale(scores3)
    scores4 = preprocessing.scale(scores4)
    scores6 = preprocessing.scale(scores6)


    best_i_j_k = (0.20000000000000284, 1.0000000000000036, 2.100000000000005)
    get_acc(scores1 * best_i_j_k[0] + scores2 * best_i_j_k[1] + scores3 * best_i_j_k[2], correct_answers,
            'EventComp + PairLSTM + SGNN ')

    best_i_j_k = (0.20000000000000284, -0.09999999999999742, 2.0000000000000044)

    get_acc(scores1 * best_i_j_k[0] + scores2 * best_i_j_k[1] + scores4 * best_i_j_k[2], correct_answers,
            'EventComp + PairLSTM + MCer ')



