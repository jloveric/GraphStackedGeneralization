from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import math as math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import pickle
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import ray
from abc import ABC, abstractmethod
from graphMapping import *

class ModelBase(ABC) :

    @abstractmethod
    def fit(self, X, y) :
        pass

    @abstractmethod
    def predict(self, X) :
        pass

    @abstractmethod
    def score(self, X, y) :
        pass

    @abstractmethod
    def computeScore(self, predicted, actual) :
        pass

    @abstractmethod
    def set_params(self) :
        pass
    
    @abstractmethod
    def getIncorrect(self,predicted, actual):
        pass

    def clone(self) :
        return deepcopy(self)


#@ray.remote
class PolynomialRegression(ModelBase) :
    def __init__(self, nOutputs, tolerance=1.0e-2) :
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

# I want on model per class, one against all.  You can think of this
# as having on neuron compute the probability for a single class
# which I think is what I want - instead of one computing for every class.
#@ray.remote
class PolynomialClassification(ModelBase) :
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

class RandomForest(ModelBase) :
    def __init__(self, nClasses, maxDepth=None, n_jobs=None, n_estimators=10) :
        #Do something here
        self.models = None
        self.correct = None
        self.score = None
        self.maxDepth = maxDepth
        self.nClasses = nClasses
        self.predictArray = np.arange(0,nClasses)
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators

    #Good
    def fit(self, X, y) :
        #This should probably performed in a better location
        X = np.nan_to_num(X)

        self.models = [RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=self.maxDepth, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=self.n_estimators, n_jobs=self.n_jobs,
            oob_score=False, random_state=0, verbose=0, warm_start=False).fit(X,y)]
        
        return self

    #Good
    def predict(self, X) :
        prob = self.models[0].predict_proba(X)
        r = np.zeros((prob.shape[0], self.nClasses))
        #print('classes', self.models[0].classes_)
        r[:,self.models[0].classes_.astype(np.int)]=prob
        return r

    def score(self, X, y) :
        correct, score = scoreSet(self.models, X, y)
        self.correct = correct
        self.score = score
        return self.score

    #Good I think - predicted is the probabilities, actual is the single actual class
    def computeScore(self, predicted, actual) :
        return computeScore(predicted, actual)

    def set_params(self) :
        #something
        return
    
    #good
    def getIncorrect(self,predicted, actual):
        #something
        return returnFailed(predicted, actual)

class Convolutional2D(ModelBase) :

    def __init__(self, modelPrototype, width, height, sampleWidth, inputStride, outputStride, maxSamples=None) :
        self.modelPrototype = modelPrototype
        self.models = None
        self.mapping = None
        self.width = width
        self.height = height
        self.sampleWidth = sampleWidth
        self.inputStride = inputStride
        self.outputStride = outputStride
        self.thisModel = None
        self.numModels = None
        self.maxSamples =  maxSamples

    def fit(self, X, y) :
        #print('fit.shape in', X.shape)
        #First create the new form of the training examples - could be a very long time
        newSamples, newLabels = createTrainingSamples2Dfrom1D(self.width, self.height, self.sampleWidth, self.inputStride, self.maxSamples, X, y,dtype=np.float16)
        self.thisModel = self.modelPrototype.clone()

        #print('newSamples.shape', newSamples.shape, 'newLabels.shape', newLabels.shape)
        self.thisModel.fit(newSamples,newLabels)
        self.models = self.thisModel.models

    def predict(self, X) :
        #print('X.shape',X.shape)
        if self.mapping is None :
            self.channels = int(X.shape[1]/(self.width*self.height))
            self.mapping = createInput2DMapping(self.width, self.height, self.outputStride, self.sampleWidth, self.channels)
            self.numModels = self.mapping.shape[0]
       

        #print('mapping.shape', self.mapping.shape)
        return applyModel(X, self.mapping, self.thisModel,dtype=np.float16)

    def score(self, X, y) :
        correct, score = scoreSet(self.models, X, y)
        self.correct = correct
        self.score = score
        return self.score 

    def computeScore(self, predicted, actual) :
        correct = (actual.shape[0]-np.sum(self.getIncorrect(predicted,actual)))
        score  = correct/actual.shape[0]
        return [correct, score]

    def set_params(self, X, y) :
        pass

    def getIncorrect(self, predicted, actual) :

        #Should actually do an average for each 49 groups
        reduced = []
        numClasses = int(predicted.shape[1]/self.numModels)
        #print('numClasses', numClasses)
        shape = predicted.shape
        temp = predicted.reshape((shape[0], self.numModels, numClasses))
        
        reduced = np.mean(temp,axis=1)
        
        #print('reduced.shape', reduced.shape)
        #reduced = np.array(reduced)
        #print('actual.shape', actual.shape, 'predicted.shape', predicted.shape)
        #print('reduced.shape', reduced.shape)
        best=np.argmax(reduced, axis=1)
        pair = np.column_stack(tuple([best,actual]))
        condition = (pair[:,0] != pair[:,1])
        return condition #np.extract(condition, np.range())


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

def classify(regSets, images,dtype=np.float16) :
    #print('images.dtype', images.dtype)

    classifications = []
    for i in range(0, len(regSets)):
        ans = regSets[i].predict(images)   
        classifications.append(ans.astype(dtype))

    final = np.column_stack(tuple(classifications)).astype(dtype)
    #print('final.dtype', final.dtype)
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
    #print('final.shape', final.shape, 'best.shape',best.shape,'labels',labels.shape)

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

