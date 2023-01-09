import itertools as it
import math
import multiprocessing as mp
from pprint import pprint
from random import randint

import numpy as np
import pandas as pd

'''
https://homel.vsb.cz/~kro080/PAI-2022/U2/

https://youtrack.jetbrains.com/issue/PY-52273/Debugger-multiprocessing-hangs-pycharm-2021.3
https://youtrack.jetbrains.com/issue/PY-37366/Debugging-with-multiprocessing-managers-no-longer-working-after-update-to-2019.2-was-fine-in-2019.1#focus=Comments-27-4690501.0-0

'''

mnist_data_file = 'mnist/mnist_train.csv'
mnist_test_file = 'mnist/mnist_test.csv'

pool_size_l1 = 4  # TODO not used
pool_size_l2 = 8

width = 28
height = 28

threshold_const = 1.5
classes_cnt = 10
training_pts_cnt_per_class = 20
iter_cnt = 50
data_cnt = 500

threshold_multiplier_const = None  # not used anymore

def load_data(filename):
    print("Load file at: ", mnist_data_file)
    nrows = data_cnt
    data = pd.read_csv(filename, nrows=nrows)
    return data

class Pt(object):
    '''
    Point
    '''
    ndim = 2
    def __init__(s, _x, _y):
        s.x = _x
        s.y = _y
        s.size = np.sqrt(s.x*s.x + s.y*s.y)

    def set(s, pt):
        s.x = pt.x
        s.y = pt.y


def convert_sum2pts(sum):
    '''
    Because IDK how to easily to get Points (Vectors) from MNIST I am converting summed data per class to points if the
    value is above given threshold,
    The other way would be to recalc for each MNIST row Feature Vector...
    ... and I am too lazy to deal with that, because this project should test
    Parallel implementation of MeanShift alg. and not Clustering
    '''
    ar = np.array(sum).reshape(width, height)
    nr = np.unravel_index(ar.argmax(), ar.shape)
    threshold = ar[nr[0]][nr[1]]/threshold_const
    pts = list()
    for x_i in range(0, width-1):
        for y_i in range(0, height - 1):
            if ar[x_i][y_i] > threshold:
                if threshold_multiplier_const is None:
                    pts.append(Pt(x_i, y_i))

    return pts

def round_pt_up(pt):
    x = 0 if np.isnan(pt.x) else math.ceil(pt.x)
    y = 0 if np.isnan(pt.y) else math.ceil(pt.y)
    x = width-1 if (x >= width) else x
    y = height-1 if (y >= height) else y
    x = 0 if (x < 0) else x
    y = 0 if (y < 0) else y
    return Pt(x, y)

def K_a(a_vec):
    ro = 1
    n = a_vec.ndim
    a_len = a_vec.size
    top = np.power(np.euler_gamma, - (np.power(a_len, 2)/(2.0 * np.power(ro, 2))))
    bottom = np.power(np.sqrt(np.pi * 2.0) * ro, n)
    K_a_res = top / bottom
    return K_a_res

def single_K(xi, x):
    K_input = Pt(xi.x - x.x, xi.y - x.y)
    K_res = K_a(K_input)
    return K_res

def eval_mean_shift_step(Apt, pts):
    '''
    Requierments: use Gauss kernel
    https://stats.stackexchange.com/questions/61743/understanding-the-mean-shift-algorithm-with-gaussian-kernel

    In our case is searched space 2D 28x28 PIXELS ...
    '''

    ret = None

    # budeme pracovat ve sdilene pameti
    with mp.Manager() as manager:
        lock = manager.Lock()
        # min val and empty permutation
        triplet = manager.Value('d', [0.0, 0.0, 0.0])
        # spustime ve vice procesech
        with mp.Pool(processes=pool_size_l2) as pool:
            # kazde vlakno v poolu se stara o cast vypoctu, rozdeleno dle seznamu 'pts'
            ret = pool.starmap(worker, zip(it.repeat(Apt), pts, it.repeat(triplet), it.repeat(lock)))

    def sum_i(ar, idx):
        sum_m = 0
        for r in ar:
            sum_m = sum_m + r[idx]
        return sum_m

    top_sum_x = sum_i(ret, 0)
    top_sum_y = sum_i(ret, 1)
    bottom_sum = sum_i(ret, 2)

    new_Apt = round_pt_up(Pt(top_sum_x/bottom_sum, top_sum_y/bottom_sum))
    return new_Apt

def worker(pt_k, pt_i, part_sums, lock):
    single_K_val = single_K(pt_i, pt_k)
    with lock:
        top_sum_x = part_sums.value[0]
        top_sum_y = part_sums.value[1]
        bottom_sum = part_sums.value[2]
        part_sums.value = [top_sum_x + (single_K_val * pt_i.x), top_sum_y + (single_K_val * pt_i.y), bottom_sum + single_K_val]

    return part_sums.value

def train(data):
    ## Possible paral.: coudl be also with batch evaluation (and then committee or merging model)
    pprint("Loading data...")
    # pprint(data)
    ## Possible paral.: sum numbers per class (10 classes -> effective 2 threads)
    # for each class
    # sum all numbers
    sums = prepare_training_data(classes_cnt, data)
    print("Training Pts...")
    training_pts_per_class = train_pts(iter_cnt, sums, training_pts_cnt_per_class)
    print("Trained...")
    return training_pts_per_class


def train_pts(iter_cnt, sums, training_pts_cnt_per_class):
    training_pts_per_class = dict()
    ## Possible paral.: sum numbers per class (10 classes -> effective 2 threads)
    # for each summed class
    for sum_i in range(0, len(sums)):
        print("sum_i: ", sum_i)
        sum = sums[sum_i]
        if sum is None:
            continue
        ## Possible paral.: per training points (4 training points -> effective 2,4 threads)
        # gen N training points (random or edges of image?)
        # for each training point t
        # training_pts = [Pt(0, 0) for x in range(0, training_pts_cnt_per_class)]
        training_pts = [Pt(randint(0, width-1), randint(0, height-1)) for x in range(0, training_pts_cnt_per_class)]
        # so for testing purposes lets generate points in corners ...
        # training_pts = [Pt(0, 0), Pt(width-1, 0), Pt(0, height-1), Pt(width-1, height-1)]
        for pt_i in range(0, training_pts_cnt_per_class):
            if pt_i % (training_pts_cnt_per_class / 10) == 0:
                print("Pt: ", pt_i," of ", training_pts_cnt_per_class)
            pt = training_pts[pt_i]
            old_pt = pt
            for it_i in range(0, iter_cnt):
                # run Mean Shift N times for t and thus move it to the nearest center of mass
                new_pt = eval_mean_shift_step(old_pt, convert_sum2pts(sum))
                old_pt = new_pt
            pt.x = new_pt.x
            pt.y = new_pt.y
            pt.size = new_pt.size
        training_pts_per_class[sum_i] = training_pts
    return training_pts_per_class


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

        pprint(i)
        # pprint(sum)
        # sums[i] = sum
    # pprint(sums)
    return sums

class TrainedModel(object):
    '''
    Traning points for each class
    '''
    training_pts = [[]]

def eval_model(model, test_data):
    for pt_i in model.keys():
        # pprint(model[pt_i])
        pts = model[pt_i]
        print("i:", pt_i)
        for pt in pts:
            # pprint(pt.x, pt.y, pt.size)
            print("    ", pt.x, pt.y, pt.size)
            # pprint(pt)

    data = list()
    for index, row in test_data.iterrows():
        row_data = list(row.copy().array)
        data.append([row_data[0], row_data[1:]])

    res = dict()
    # for data_i in range(0, len(model)-1):
    for data_i in range(0, len(data)):
        best_found = 0
        best_found_i = None
        # if data_i not in data.keys():
        #     continue
        img1 = data[data_i][1]
        label = data[data_i][0]
        img = np.array(img1).reshape(width, height)
        for m_i in range(0, len(model)):
            if m_i not in model.keys():
                continue
            best_found_pt_ii = list()
            for pt_i in range(0, len(model[m_i])):
                pt = model[m_i][pt_i]
                val = list(img)[pt.x][pt.y]
                if val >= best_found:
                    best_found_pt_ii.append([pt, val, m_i, pt_i])
                    # best_found = val
                    # best_found_i = m_i
            best_found_pt_ii = sorted(best_found_pt_ii, key=lambda bf: -bf[1])
            if len(best_found_pt_ii) > 0 and best_found_pt_ii[0][1] >= best_found:
                best_found_i = best_found_pt_ii[0][2]
                best_found = best_found_pt_ii[0][1]
        res[data_i] = [label, best_found_i]

    return res

def eval_res(res):
    print("eval_res:")
    pprint(res)
    ok = 0
    for k in res.keys():
        r = res[k]
        if r[0] == r[1]:
            ok = ok + 1

    print("ok: " + str(ok) + " of " + str(len(res)) + ": " + str(ok/len(res)))

def run():
    data = load_data(mnist_data_file)
    model = train(data)
    test_data = load_data(mnist_test_file)
    res = eval_model(model, test_data)
    eval_res(res)

run()
