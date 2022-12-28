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
    nrows = 20
    data = pd.read_csv(filename, nrows=nrows)
    return data

def train(data):
    ## Paral.: coudl be also with batch evaluation (and then committee or merging model)
    pprint(data)
    ## Paral.: sum numbers per class (10 classes -> effective 2 threads)
    classes_cnt=10
    sums = [None for x in range(0,classes_cnt)] # define array of sums
    for i in range(0, classes_cnt-1):
        sum = None
        for index, row in data.iterrows():
            row_data = list(row.array)
            sum_i = row_data[0]
            if sums[sum_i] is None:
                row_data = list(row.copy().array)
                sum_i = row_data[0]
                sums[sum_i] = row_data[1:]
            else:
                row_data = list(row.array)
                sum_i = row_data[0]
                for i in range(1, len(row_data)-1):
                    sums[sum_i][i] = sums[sum_i][i] + row_data[i]

        pprint(sum)
        sums.append(sum)
    pprint(sums)
    # for each class
    # sum all numbers
    ## Paral.: sum numbers per class (10 classes -> effective 2 threads)
    # for each summed class
    ## Paral.: per traning points (4 training points -> effective 2,4 threads)
    # gen N training points (random or edges of image?)
    # for each training point t
    # run Mean Shift N times for t and thus move it to the nearest center of mass

class TrainedModel(object):
    '''
    Traning points for each class
    '''
    training_pts = [[]]

def eval_model(test_data, model):
    # return classification of test data
    return None

def run():
    data = load_data('mnist_train.csv')
    model = train(data)
    test_data = load_data('mnist_test.csv')
    res = eval_model(model, test_data)
    pprint(res)
run()
