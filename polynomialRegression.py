import numpy as np
import multiclassRegression as mr 
from sklearn import preprocessing
import polynomial
import ray
from iteration_utilities import deepflatten

def expand(data, a) :
    nImages = a(data)
    nImages = nImages.reshape(nImages.shape[0],nImages.shape[1]*nImages.shape[2])
    return nImages

@ray.remote(num_return_vals=1)
def classify(model, data) :
    return mr.classify(model,data) 

@ray.remote(num_return_vals=3)
def computeModelSet(nextData, nextLabels, modelsInLayer, p, index, lastLayer=False, metric=None) :
    totalSize = nextData.shape[0]
    numFailed = totalSize
    totalFailures = 0

    if (modelsInLayer > 1) and (lastLayer!=True) :
        randomSet = np.random.choice([True, False],totalSize,p=[p,(1.0-p)],replace=True)
        thisData = nextData[randomSet]
        thisLabels = nextLabels[randomSet]
    else :
        thisData = nextData
        thisLabels = nextLabels
        print('shapes', thisData.shape, thisLabels.shape)

    modelSet = []
    while numFailed!=0 :
        
        metric.fit(thisData, thisLabels)
        model = metric.models

        modelSet.append(model)

        final = metric.predict(thisData)
        correct, score = metric.computeScore(thisData, thisLabels)
        failed = metric.getIncorrect(final, thisLabels)
        numFailed = np.sum(failed)

        print('score', score, 'failed', numFailed, 'number', len(modelSet), 'id', index)
        
        #In the last layer we don't care bout computing the error models
        if lastLayer == True :
            break

        if numFailed > 0 :
            totalFailures = totalFailures+1
            thisData = thisData[failed]
            thisLabels = thisLabels[failed]
        
    return modelSet, totalFailures, index

def buildParallel(nextData, nextLabels, modelsInLayer, modelSize, basis) :

    numLayers = len(modelsInLayer)

    totalSize = nextData.shape[0]
    p = modelSize/totalSize

    allModelSets = []
    transformSet = []
    layer = 0

    nextLabelsId = ray.put(nextLabels)

    while numLayers > 0 :
        atLeastOneFailed = False
        modelSet = []
        failures = []

        nextDataId = ray.put(nextData, weakref=True)

        oldData = np.copy(nextData)
        print("Constructing layer ------------------------------", len(allModelSets))
        for i in range(0,modelsInLayer[layer]) :
            thisModelSet, totalFailures, index = computeModelSet.remote(nextDataId, nextLabelsId, modelsInLayer[layer], p, i, lastLayer = (numLayers==1), metric=mr.MultiClassRegression(nLabels=10))
            modelSet.append(thisModelSet)
            failures.append(totalFailures)

        index = ray.get(index)
        modelSet = ray.get(modelSet)
        modelSet = list(deepflatten(modelSet,depth=1))
        failures = ray.get(failures)
        atLeastOneFailed = any(failures)
        
        #print('modelSet',modelSet)
        allModelSets.append(modelSet)
        layer = layer + 1

        '''
        Build another model with the output of the previous set
        '''
        
        oldDataId = ray.put(oldData, weakref=True)

        modelSetIds = []
        for i in modelSet :
            modelSetIds.append(ray.put(i))

        print('creating new classification estimates')
        inputSet = mr.classify(modelSet[0],oldData)
        results = []
        for i in range(1,len(modelSet)) :
            #print('i', i)
            results.append(classify.remote(modelSetIds[i],oldDataId))   
        
        results = ray.get(results)
        
        #ok, this needs to be done in parallel also.
        print('stacking')
        for i in results :
            inputSet = np.dstack((inputSet, i))

        ts = inputSet.shape
        if len(ts) > 2 :
            inputSet = inputSet.reshape((ts[0],ts[1]*ts[2]))

        #reset everything
        print('rescaling')
        scaler = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(inputSet)
        inputSet = scaler.transform(inputSet)
        transformSet.append(scaler)

        print('applying expansion')
        nextData = expand(inputSet, basis)
        nextLabels = nextLabels
        numLayers = numLayers -1

        if atLeastOneFailed == False :
            print('exiting early since there were no failures on submodels')
            break

        print('next input shape', nextData.shape)

    return allModelSets, transformSet, basis

def evaluateModel(nextData, labels, allModelSets, transformSet, basis) :

    numLevels = len(allModelSets)
    for level in range(0, numLevels) :
        print("Computing output for layer",level+1)
        theseModels = allModelSets[level]
        numModels = len(theseModels)
        
        inputSet = mr.classify(theseModels[0],nextData)
        correct, percent = mr.computeScore(inputSet, labels)
        print('correct', correct, 'fraction correct', percent)
        
        for model in range(1, numModels) :
            res = mr.classify(theseModels[model],nextData)
            inputSet = np.dstack((inputSet, res))
            correct, percent = mr.computeScore(res, labels)
            print('correct', correct, 'fraction correct', percent)

        ts = inputSet.shape
        print('ts', ts)
        if len(ts) > 2 :
            inputSet = inputSet.reshape((ts[0],ts[1]*ts[2]))
        inputSet = transformSet[level].transform(inputSet)
        nextData = expand(inputSet, basis)