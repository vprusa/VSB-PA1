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

    ndim = 2

    def __init__(s, _x, _y):
        s.x = _x
        s.y = _y
        s.tp_range = 10
        s.size = np.sqrt(s.x*s.x + s.y*s.y)

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

def convert_sum2pts(sum):
    '''
    Because IDK how to easily to get Points (Vectors) from MNIST I am converting summed data per class to points if the
    value is above given threshold,
    The other way would be to recalc for each MNIST row Feature Vector...
    ... and I am too lazy to deal with that, because this project should test
    Parallel implementation of MeanShift alg. and not Clustering
    '''
    ar = np.array(sum).reshape(28, 28)
    nr = np.unravel_index(ar.argmax(), ar.shape)
    threshold = nr/2.0
    pts = list()
    for x_i in range(0,width-1):
        for y_i in range(0, height - 1):
            if ar[x_i][y_i] > threshold:
                pts.append(Pt(x_i, y_i))
    return pts


def eval_mean_shift_step_K(pt):
    return Pt()

def get_in_range(pt, sum):
    for i in range(0, len(sum)):
        sum_at()

def eval_mean_shift_step(Apt, pts):
    # np.array(sum).reshape(28,28)
    # centers = [pt.x, pt.y]
    # X, _ = make_blobs(n_samples=950, centers=centers, cluster_std=0.89)
    # ms = MeanShift()
    # ms.fit(np.array(sum).reshape(28,28))
    # lets cheat ... TODO do not cheat
    # if sum is None:
    #     return pt
    # ar = np.array(sum).reshape(28, 28)
    # nr = np.unravel_index(ar.argmax(), ar.shape)
    # pt.x = nr[0]
    # pt.y = nr[1]

    '''
    Requierments: use Gauss kernel
    https://stats.stackexchange.com/questions/61743/understanding-the-mean-shift-algorithm-with-gaussian-kernel

    In our case is searched space 2D 28x28 PIXELS ...
    '''

    def K_a(a_vec):
        ro = 10
        n = a_vec.ndim
        a_len = a_vec.size
        top = np.power(np.euler_gamma, - (np.power(a_len, 2)/(2.0* np.power(ro,2))))
        bottom = np.power(np.sqrt(np.PI * 2.0) * ro, n)
        K_a_res = top / bottom
        return K_a_res

    def single_K(xi, x):
        K_input = Pt(xi[0] - x[0], xi[1] - x[1])
        K_res = K_a(K_input)
        return K_res

    top_sum_x = 0
    top_sum_y = 0
    bottom_sum = 0
    for pti in pts:
        single_K_val = single_K(pti, Apt)
        top_sum_x = single_K_val * Apt.x
        top_sum_y = single_K_val * Apt.y
        bottom_sum = bottom_sum + single_K_val

    new_Apt = Pt(top_sum_x/bottom_sum, top_sum_y/bottom_sum)
    return new_Apt

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
                new_pt = eval_mean_shift_step(pt, convert_sum2pts(sum))
                pt.x = new_pt.x
                pt.y = new_pt.y
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
