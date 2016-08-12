import os
import csv

def GetRNCData():
    data = []

    with open('rnc-speech', 'r') as f:
        for line in f:
            data.append(line)

    return data

def GetDNCData():
    data = []

    with open('dnc-speech', 'r') as f:
        for line in f:
            data.append(line)

    return data

def GetIMDBData():
    train_pos = 'aclImdb/train/pos/'
    train_pos_data = []
    for file in os.listdir(train_pos):
        if file.endswith(".txt"):
            with open(train_pos + file) as f:
                train_pos_data.append(f.read())

    train_neg = 'aclImdb/train/neg/'
    train_neg_data = []
    for file in os.listdir(train_neg):
        if file.endswith(".txt"):
            with open(train_neg + file) as f:
                train_neg_data.append(f.read())

    train_unsup = 'aclImdb/train/unsup/'
    train_unsup_data = []
    for file in os.listdir(train_unsup):
        if file.endswith(".txt"):
            with open(train_unsup + file) as f:
                train_unsup_data.append(f.read())

    return train_pos_data, train_neg_data, train_unsup_data

def GetTestIMDB():
    test_pos = 'aclImdb/test/pos/'
    test_pos_data = []
    for file in os.listdir(test_pos):
        if file.endswith(".txt"):
            with open(test_pos + file) as f:
                test_pos_data.append(f.read())

    test_neg = 'aclImdb/test/neg/'
    test_neg_data = []
    for file in os.listdir(test_neg):
        if file.endswith(".txt"):
            with open(test_neg + file) as f:
                test_neg_data.append(f.read())

    return test_pos_data, test_neg_data

def GetTwitterData():
    train_pos_data = []
    train_neg_data = []

    with open('SentimentAnalysisDataset.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)
        for row in csvreader:
            if int(row[1]) == 1:
                train_pos_data.append(row[3])
            elif int(row[1]) == 0:
                train_neg_data.append(row[3])

    return train_pos_data, train_neg_data

def GetTrainData(IMDB=False, twitter=False, filename=None):
    if IMDB:
        return GetIMDBData()
    elif twitter:
        pos, neg = GetTwitterData()
        return pos[:-10000], neg[:-10000], None

def GetTestData(IMDB=False, twitter=False):
    if IMDB:
        return GetTestIMDB()
    elif twitter:
        pos, neg = GetTwitterData()
        return pos[-10000:], neg[-10000:], None

def GetUnlabeledTestData(DNC=False, RNC=False):
    if DNC:
        return GetDNCData()
    elif RNC:
        return GetRNCData()
