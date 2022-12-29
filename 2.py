import itertools as it
import multiprocessing as mp
import numpy as np
from pprint import pprint
import csv
import pandas as pd

'''
https://homel.vsb.cz/~kro080/PAI-2022/U2/

https://youtrack.jetbrains.com/issue/PY-52273/Debugger-multiprocessing-hangs-pycharm-2021.3
https://youtrack.jetbrains.com/issue/PY-37366/Debugging-with-multiprocessing-managers-no-longer-working-after-update-to-2019.2-was-fine-in-2019.1#focus=Comments-27-4690501.0-0
'''

def load_data(filename):
    # csv_reader = csv.reader(filename)
    # print(csv_reader)
    # return csv_reader
    nrows = 200
    data = pd.read_csv(filename, nrows=nrows)
    return data

class Pt(object):
    '''
    Training point
    '''

    def __init__(s, _x, _y):
        s.x = _x
        s.y = _y
        s.tp_range = 10

    def set(s, pt):
        s.x = pt.x
        s.y = pt.y


width = 28
height = 28

def sum_at(x,y):
    return

shift_step_ro = 1
shift_step_n = 2

from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs

def eval_mean_shift_step_K(pt):
    return Pt()

def get_in_range(pt, sum):
    for i in range(0, len(sum)):
        sum_at()

def eval_mean_shift_step(pt, sum):
    # np.array(sum).reshape(28,28)
    # centers = [pt.x, pt.y]
    # X, _ = make_blobs(n_samples=950, centers=centers, cluster_std=0.89)
    # ms = MeanShift()
    # ms.fit(np.array(sum).reshape(28,28))
    # lets cheat ... TODO do not cheat
    if sum is None:
        return pt
    ar = np.array(sum).reshape(28, 28)
    nr = np.unravel_index(ar.argmax(), ar.shape)
    pt.x = nr[0]
    pt.y = nr[1]
    return pt

def train(data):
    ## Paral.: coudl be also with batch evaluation (and then committee or merging model)
    pprint(data)
    ## Paral.: sum numbers per class (10 classes -> effective 2 threads)
    # for each class
    # sum all numbers
    classes_cnt=10
    sums = prepare_training_data(classes_cnt, data)
    training_pts_cnt_per_class = 1
    iter_cnt = 2
    training_pts_per_class = dict()

    train_pts(iter_cnt, sums, training_pts_cnt_per_class, training_pts_per_class)
    return training_pts_per_class


def train_pts(iter_cnt, sums, training_pts_cnt_per_class, training_pts_per_class):
    ## Paral.: sum numbers per class (10 classes -> effective 2 threads)
    # for each summed class
    for sum_i in range(0, len(sums)):
        sum = sums[sum_i]
        ## Paral.: per training points (4 training points -> effective 2,4 threads)
        # gen N training points (random or edges of image?)
        # for each training point t
        training_pts = [Pt(0, 0) for x in range(0, training_pts_cnt_per_class)]
        for pt_i in range(0, training_pts_cnt_per_class):
            pt = training_pts[pt_i]
            for it_i in range(0, iter_cnt - 1):
                # run Mean Shift N times for t and thus move it to the nearest center of mass
                eval_mean_shift_step(pt, sum)
        training_pts_per_class[sum_i] = training_pts


def prepare_training_data(classes_cnt, data):
    # sums = [None for x in range(0, classes_cnt)] # define array of sums
    sums = dict()  # define array of sums
    for i in range(0, classes_cnt - 1):
        sum = None
        for index, row in data.iterrows():
            row_data = list(row.array)
            sum_i = row_data[0]
            if sum_i not in sums.keys() or sums[sum_i] is None:
                row_data = list(row.copy().array)
                sum_i = row_data[0]
                sums[sum_i] = row_data[1:]
            else:
                row_data = list(row.array)
                sum_i = row_data[0]
                for ii in range(1, len(row_data) - 1):
                    sums[sum_i][ii] = sums[sum_i][ii] + row_data[ii]

        pprint(sum)
        sums[i] = sum
    pprint(sums)
    return sums


class TrainedModel(object):
    '''
    Traning points for each class
    '''
    training_pts = [[]]

def eval_model(model, test_data):

    data = dict()
    for index, row in test_data.iterrows():
        row_data = list(row.copy().array)
        data[row_data[0]] = row_data[1:]

    res = dict()
    for data_i in range(0, len(data)):
        best_found = 0
        best_found_i = None
        if data_i not in data.keys():
            continue
        img1 = data[data_i]
        img = np.array(img1).reshape(28, 28)
        for m_i in range(0, len(model)):
            pt = model[m_i][0]
            if img[pt.x][pt.y] > best_found:
                best_found = img[pt.x][pt.y]
                best_found_i = m_i
            res[data_i] = best_found_i

    return res

def run():
    data = load_data('mnist_train.csv')
    model = train(data)
    test_data = load_data('mnist_test.csv')
    res = eval_model(model, test_data)
    pprint(res)
run()
