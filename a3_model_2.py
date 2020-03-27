import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from FFNN import FFNNModel
# Whatever other imports you need


# You can implement classes and helper functions here too.
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))


def read_in_csv(filename):
    with open(filename, "r") as thefile:
        data = pd.read_csv(thefile)
    return data


def sample_data(size, df):
    def compare_author(i1, i2):
        d1_df = df[df.ID == i1]
        d2_df = df[df.ID == i2]
        author1 = d1_df['0']
        a1 = author1.values.tolist()
        author2 = d2_df['0']
        a2 = author2.values.tolist()
        d1 = d1_df.values.tolist()
        d1 = d1[0][3:]
        d2 = d2_df.values.tolist()
        d2 = d2[0][3:]
        if a1 == a2:
            c = 1
        else:
            c = 0
        d = d1 + d2
        sample = (d, c)
        return sample

    samples = []
    counter1 = 0
    counter0 = 0
    while len(samples) <= size:
        index1 = random.choice(df.index)
        index2 = random.choice(df.index)
        while index1 == index2:
            index2 = random.choice(df.index)
        sample = compare_author(index1, index2)
        if sample[1] == 1:
            if counter1 <= size/2 + 1:
                samples.append(sample)
                counter1 += 1
        else:
            if counter0 <= size/2 + 1:
                samples.append(sample)
                counter0 += 1
    random.shuffle(samples)
    return samples


def sample_testdata(size, test_df):
    samples = sample_data(size, test_df)
    labels = [x[1] for x in samples]
    testdata = [torch.Tensor(x[0]) for x in samples]
    return testdata, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("--size", "-S", dest="samplesize", type=int, default="100", help="Size of the sample.")
    parser.add_argument("--hidden", "-H", dest="hiddenlayersize", type=int, default="0",
                        help="Size of the hiddenlayer.")
    parser.add_argument("--nonlin", "-L", dest="nonlinearity", type=str, default="",
                        help="Name of the nonlinearity.")
    parser.add_argument("--out", "-O", dest="outputfile", type=str, default="",
                        help="The name of the output file containing the plotted results.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile, args.samplesize))
    # implement everything you need here
    data = read_in_csv(args.featurefile)
    train_df = data[data.train_test == 'train']
    test_df = data[data.train_test == 'test']
    samples = sample_data(args.samplesize, train_df)
    testdata, labels = sample_testdata(args.samplesize, test_df)
    inputsize = len(samples[0][0])
    hiddensize = args.hiddenlayersize
    nonlinearity = args.nonlinearity
    hiddensizes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    precisions = []
    recalls = []
    for hiddensize in hiddensizes:
        model = FFNNModel(input_size=inputsize, hidden_size=hiddensize, non_lin=nonlinearity)
        model.train_model(samples)
        results = model.test(testdata, labels)
        precision, recall = results[1], results[2]
        precisions.append(precision)
        recalls.append(recall)
    df = pd.DataFrame({'recall': recalls, 'precision': precisions, 'hiddensize': hiddensizes})
    df = df.sort_values('recall')
    lines = df.plot(x='recall', y='precision', kind='line').get_figure()
   # df[['recall', 'precision', 'hiddensize']].apply(lambda row: lines.text(*row), axis=1);
    lines.savefig(args.outputfile)



