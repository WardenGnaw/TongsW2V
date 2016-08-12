import sys
import os
import gensim
import logging
import pickle
import numpy

class TONGSWord2Vec:
    def __init__(self, data=None, filename=None):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
        level=logging.INFO)

        if data == None and filename == None:
            if os.path.exists('GoogleNews-vectors-negative300.bin.gz'):
                self.model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
            else:
                print('No data inputted and GoogleNews-vectors-negative300.bin.gz does not exist.')
                sys.exit(1)
        else:
            if data:
                self.model = gensim.models.Word2Vec(data, size=300, window=5, min_count=10, workers=4, iter=20)
            elif filename:
                self.model = gensim.models.Word2Vec.load(filename)

    def Save(self, filename):
        self.model.save(filename)

    def ConvertSentenceToVector(self, sentence_vec, norm=False):
        ret = []
        for word in sentence_vec:
            try:
                ret.append(self.model[word])
            except KeyError:
                pass

        ret = numpy.sum(ret, axis=0)

        if norm:
            # Removed normalization of vectors because it lowered accuracy
            return sklearn.preprocessing.normalize(ret.reshape(1, -1), axis=1, norm='l2')[0]
        else:
            return ret

    def ConvertSentencesToVectors(self, sentences, norm=False):
        ret = []

        for sentence in sentences:
            ret.append(self.ConvertSentenceToVector(sentence, norm))

        return ret

    def PMI(self, word):
        return self.model.n_similarity([word], ['excellent']) - self.model.n_similarity([word], ['poor'])

    def SentencePMI(self, sentence):
        PMI_total = 0

        for word in sentence:
            try:
                PMI_total += self.PMI(word)
            except KeyError:
                pass

        return PMI_total
            

