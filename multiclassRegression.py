from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import math as math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from joblib import dump, load
import pickle
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import ray

#@ray.remote
class MultiOutputRegression :
    def __init__(self, nOutputs, tolerance=1.0e-3) :
        self.models = None
        self.correct = None
        self.score = None
        self.nOutputs = nOutputs
        self.tolerance = tolerance

    #Good
    def fit(self, X, y) :
        self.models = multiClassRegression(X, y.transpose())
        return self

    #Good
    def predict(self, X) :
        return classify(self.models, X)

    def score(self, X, y) :
        #Now do a metric distance, the smaller the better?
        out = self.predict(X)
        correct, score = self.computeScore(out, y)
        self.score = score
        return self.score

    #Good
    def computeScore(self, predicted, actual) :

        correct = np.where(np.abs(predicted-actual)<=self.tolerance, True, False)

        #diff = np.linalg.norm(predicted-actual)
        #diff = diff/diff.size
        numCorrect = np.sum(correct)

        return np.sum(correct), numCorrect/correct.size

    def set_params(self) :
        #something
        return

    def getIncorrect(self,predicted, actual) :
        #something
        incorrect = np.where(np.abs(predicted-actual)>self.tolerance, True, False)
        return incorrect.flatten()

    def clone(self) :
        #I don't actually want to clone the models - this is just to get the same initial conditions
        return deepcopy(self)

# I want on model per class, one against all.  You can think of this
# as having on neuron compute the probability for a single class
# which I think is what I want - instead of one computing for every class.
#@ray.remote
class MultiClassClassification :
    def __init__(self, nLabels) :
        #Do something here
        self.models = None
        self.correct = None
        self.score = None
        self.nLabels = nLabels

    def fit(self, X, y) :
        self.models = curveFit(X, y, self.nLabels)
        return self

    #@ray.method(num_return_vals=1)
    def predict(self, X) :
        return classify(self.models, X)

    def score(self, X, y) :
        correct, score = scoreSet(self.models, X, y)
        self.correct = correct
        self.score = score
        return self.score

    def computeScore(self, predicted, actual) :
        return computeScore(predicted, actual)

    def set_params(self) :
        #something
        return

    def getIncorrect(self,predicted, actual):
        #something
        return returnFailed(predicted, actual)

    def clone(self) :
        #I don't actually want to clone the models - this is just to get the same initial conditions
        return deepcopy(self)

def multiClassRegression(data, labelSets, filename=None) :

    regSet = []
    
    if labelSets.shape[0]==1 or len(data.shape)==1 :
        regSet.append(Ridge(alpha=0.001, tol=0.001).fit(data, labelSets.flatten()))
    else :
        for i in range(0,labelSets.shape[0]) :
            #reg = LinearRegression().fit(data, labelSets[i])
            reg = Ridge(alpha=0.5, tol=0.001).fit(data, labelSets[i])
            regSet.append(reg)
    
    '''
    for i in range(0,len(labelSets)) :
        #reg = LinearRegression().fit(data, labelSets[i])
        reg = Ridge(alpha=0.5, tol=0.001).fit(data, labelSets[i])
        regSet.append(reg)
    '''
    return regSet

#This assumes your labels are 0...n-1
#and for every class returns 1 or 0 as
#to whether it belongs to.  It returns
#n label sets
def booleanSingleClassLabel(labels,n) :
    labelSets = []
    for i in range(0,n) :
        arr = np.array(labels)
        b = np.where(arr!=i,0,1)
        
        labelSets.append(b)
    return np.array(labelSets)

def classify(regSets, images) :

    classifications = []
    for i in range(0, len(regSets)):
        ans = regSets[i].predict(images)   
        classifications.append(ans)

    final = np.column_stack(tuple(classifications))

    return final

#Single best guess (largest value)
def bestGuess(final) :
    best=np.argmax(final, axis=1)
    return best

#closest in distance to 1.0
def nearestGuess(final) :
    closest = []
    for i in range(0, final.shape[0]) :
        vec = final[i]
        tMax = abs(vec[0]-1.0)
        idMax = 0
        for j in range(1, vec.shape[0]) :
            newDiff = abs(vec[j]-1.0)
            if(newDiff<tMax) :
                tMax=newDiff
                idMax=j
        closest.append(idMax)
    return np.array(closest)


'''
Assumes the original data before one hot encoding was encoded
as 0 to n-1
'''
def computeScore(final, labels, dumpError=False) :
    best=np.argmax(final, axis=1)
    #print('best.shape',best.shape)

    pair = np.column_stack(tuple([best, labels]))
    #print('best',pair)
    correct = 0
    for i in range(0,pair.shape[0]) :
        if pair[i][0]==pair[i][1] :
            correct=correct+1
        else :
            if dumpError :
                print(i,pair[i][0],pair[i][1],final[i])
    
    return [correct, correct/pair.shape[0]]

def returnFailed(final, labels) :
    best=np.argmax(final, axis=1)
    pair = np.column_stack(tuple([best,labels]))
    condition = (pair[:,0] != pair[:,1])
    return condition #np.extract(condition, np.range())

def curveFit(data, labels, nLabels, filename=None):
    labelSets = booleanSingleClassLabel(labels,nLabels)
    regTrain = multiClassRegression(data, labelSets, filename)
    return regTrain

def finalAndScore(regression, data, labels):
    final = classify(regression,data)
    [correct, score] = computeScore(final, labels)
    return score, final

def scoreSet(regression,data, label) :
    return finalAndScore(regression, data, label)[0]

