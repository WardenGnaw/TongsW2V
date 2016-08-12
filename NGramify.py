from nltk.util import ngrams

def ExtendDatasetBigrams(dataset):
    new_dataset = []

    for data in dataset:
        new_dataset.append(data)
        for i in range(len(data) - 1):
            sentence = (' '.join(data[:i]) + ' ' + data[i] + '_' + data[i + 1] + ' ' + ' '.join(data[i+2:])).strip().split(' ')
            new_dataset.append(sentence)

    return new_dataset

