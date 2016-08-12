import sys
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
import pickle

class TONGSClassifierError(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)

class TONGSClassifier:
    def __init__(self, name, rebuild=True, SVM=False, RF=False, NB=False, LR=False):
        self.clf = None

        if rebuild:
            if SVM:
                self.clf = svm.SVC(kernel="poly", degree=2, verbose=True)
                # param = {'C': [1e15,1e13,1e11,1e9,1e7,1e5,1e3,1e1,1e-1,1e-3,1e-5]}
                # self.clf = GridSearchCV(self.clf, param, cv=10)
                self.name = "{0}.svm".format(name)
            elif RF:
                self.clf = RandomForestClassifier(n_estimators=2)
                self.name = "{0}.rf".format(name)
            elif NB:
                self.clf = GaussianNB()
                self.name = "{0}.nb".format(name)
            elif LR:
                self.clf = LogisticRegression()
                self.name = "{0}.lr".format(name)
        else:
            if SVM:
                self.name = "{0}.svm".format(name)
            elif RF:
                self.name = "{0}.rf".format(name)
            elif NB:
                self.name = "{0}.nb".format(name)
            elif LR:
                self.name = "{0}.lr".format(name)


            try:
                with open(self.name, 'rb') as f:
                    self.clf = pickle.load(f)
            except FileNotFoundError as err:
                print("File not found error: {0}".format(err))
                sys.exit(1)

        if self.clf == None:
            print("Classifier not chosen, please pick SVM, RF, NB, or LR")
            sys.exit(1)

    def train(self, X, y):
        self.clf.fit(X, y)

    def classify(self, X):
        return self.clf.predict(X.reshape(1, -1))

    def save(self):
        with open(self.name, 'wb') as f:
            pickle.dump(self.clf, f)
