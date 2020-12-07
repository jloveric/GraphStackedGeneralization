import numpy as np
import matplotlib.pyplot as plt
import math as math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from joblib import dump, load
import pickle
from sklearn import preprocessing

def multiClassRegression(images, labelSets, filename=None) :

    regSet = []
    for i in range(0,len(labelSets)) :
        #print('npImages.shape',images.shape)
        #print('labelsSets[i].shape',labelSets[i].shape)
        #reg = LinearRegression().fit(images, labelSets[i])
        reg = Ridge(alpha=0.5, tol=0.001).fit(images, labelSets[i])
        regSet.append(reg)

    if filename!=None :
        dump(regSet,filename)

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
    return labelSets

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

def finalAndScore(regression, data, labels ):
    final = classify(regression,data)
    [correct, score] = computeScore(final, labels)
    #print('correct',correct,score)
    return score, final

def scoreSet(regression,data, label) :
    return finalAndScore(regression, data, label)[0]

