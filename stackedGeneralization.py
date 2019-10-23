import numpy as np
import multiclassRegression as mr 
from sklearn import preprocessing
import polynomial
import ray
from iteration_utilities import deepflatten
from layerData import *


def expand(data, a) :
    nImages = a(data)
    nImages = nImages.reshape(nImages.shape[0],nImages.shape[1]*nImages.shape[2])
    return nImages

@ray.remote(num_return_vals=1)
def classify(modelInfo, data) :
    #The data could be calculated outside this function so not
    #as much needs to be passed through - though it's all saved
    #elswhere so maybe it's fine.
    return modelInfo.model.predict(data[:,modelInfo.inputIndexes])

@ray.remote(num_return_vals=3)
def computeModelSet(nextData, nextLabels, modelsInLayer, p, index, lastLayer=False, metricPrototype=None,maxFailures=10) :
    totalSize = nextData.shape[0]
    numFailed = totalSize
    totalFailures = 0

    if (modelsInLayer > 1) and (lastLayer!=True) and (p!=1):
        randomSet = np.random.choice([True, False],totalSize,p=[p,(1.0-p)],replace=True)
        thisData = nextData[randomSet]
        thisLabels = nextLabels[randomSet]
    else :
        thisData = nextData
        thisLabels = nextLabels

    modelSet = []
    while numFailed!=0 :
        
        metric = metricPrototype.clone()

        metric.fit(thisData, thisLabels)
        #model = metric.models

        modelSet.append(metric)

        final = metric.predict(thisData)
        correct, score = metric.computeScore(final, thisLabels)
        failed = metric.getIncorrect(final, thisLabels)
        newFailed = np.sum(failed)

        if newFailed == numFailed :
            print('failures not improving, next.')
            break
        numFailed = newFailed

        print('score', score, 'failed', numFailed, 'number', len(modelSet), 'id', index, flush=True)
        
        #In the last layer we don't care bout computing the error models
        if lastLayer == True :
            break
        
        if totalFailures > maxFailures :
            break

        if numFailed > 0 :
            totalFailures = totalFailures+1
            thisData = thisData[failed]
            thisLabels = thisLabels[failed]
        
    return modelSet, totalFailures, index

def constructLayerInputs(layerDetails, data) :
    
    baseModels = layerDetails.numberOfBaseModels
    size = data[0].size
    if layerDetails.inputFeatures == FeatureMap.all :
        #This is overkill, but will optimize someday
        layerDetails.inputIndexes = np.array([np.arange(0,size)]*layerDetails.numberOfBaseModels)
    elif layerDetails.inputFeatures == FeatureMap.even :
        #Try to split the space evenly
        os = int(size/baseModels)
        modelInputs = [np.arange(i*os,(i+1)*os) for i in range(0,baseModels-1)]
        modelInputs.append(np.arange((baseModels-1)*os, size))
        layerDetails.inputIndexes = np.array(modelInputs)

    elif layerDetails.inputFeatures == FeatureMap.overlap :
        #overlap evenly
        print('layerDetails overlap is not implemented')

def buildParallel(nextData, nextLabels, layerDetails, modelSize, basis) :

    numLayers = len(layerDetails)

    totalSize = nextData.shape[0]
    #p = modelSize/totalSize
    p=1

    allModelSets = []
    transformSet = []
    layer = 0

    nextLabelsId = ray.put(nextLabels)
    learnerPrototypeId = []
    for i in layerDetails :
        learnerPrototypeId.append(ray.put(i.learnerPrototype))

    while numLayers > 0 :
        atLeastOneFailed = False
        modelSet = []
        failures = []

        nextDataId = ray.put(nextData, weakref=True)

        oldData = np.copy(nextData)
        print("Constructing layer ------------------------------", len(allModelSets),flush=True)

        baseModels = layerDetails[layer].numberOfBaseModels
        layerData = layerDetails[layer]

        constructLayerInputs(layerData, nextData)

        for i in range(0,baseModels) :

            thisDataId = ray.put(nextData[:,layerData.inputIndexes[i]],weakref=True)

            thisModelSet, totalFailures, index = computeModelSet.remote(thisDataId, nextLabelsId, 
                                                            layerData.numberOfBaseModels, p, i, 
                                                            lastLayer = (numLayers==1), 
                                                            metricPrototype=learnerPrototypeId[layer], #metricPrototypeId,
                                                            maxFailures = layerData.maxSubModels)

            #Ok, this might need special consideration, not sure
            modelSet.append(thisModelSet)
            
            #modelSet.append(thisModelSet)
            failures.append(totalFailures)

        index = ray.get(index)
        modelSet = ray.get(modelSet)

        #Incomprehensible comprehension
        #modelSet = [[ModelData(inputIndexes = layerData.inputIndexes[modelGroup], model = modelSet[modelGroup][i]) for i in range(0,len(modelSet[modelGroup]))] for modelGroup in range(0,len(modelSet))]
        #comprehensible version
        newModelSet = []
        for modelGroup in range(0,len(modelSet)) :
            for i in range(0, len(modelSet[modelGroup])) :
                newModelSet.append(ModelData(inputIndexes = layerData.inputIndexes[modelGroup], model=modelSet[modelGroup][i]) )

        modelSet = list(deepflatten(newModelSet,depth=1))
        failures = ray.get(failures)
        atLeastOneFailed = any(failures)
        
        #print('modelSet',modelSet)
        allModelSets.append(modelSet)
        layer = layer + 1

        '''
        Build another model with the output of the previous set
        '''
        
        oldDataId = ray.put(oldData, weakref=True)

        #Ok, this is actually correct
        modelSetIds = []
        for i in modelSet :
            modelSetIds.append(ray.put(i, weakref=True))

        print('creating new classification estimates')
        results = [classify.remote(modelSetIds[i],oldDataId) for i in range(0, len(modelSet))]
        results = ray.get(results)

        #ok, this needs to be done in parallel also.
        print('stacking')
        inputSet = results[0]
        for i in range(1,len(results)) :
            inputSet = np.dstack((inputSet, results[i]))

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
    inputSet = None
    for level in range(0, numLevels) :
        print("-------------------Computing output for layer------------------",level+1)
        theseModels = allModelSets[level]
        numModels = len(theseModels)
        
        theseIndexes = theseModels[0].inputIndexes
        inputSet = theseModels[0].model.predict(nextData[:,theseIndexes])
        correct, percent = theseModels[0].model.computeScore(inputSet, labels)
        print('correct', correct, 'fraction correct', percent)
        
        for model in range(1, numModels) :
            theseIndexes = theseModels[model].inputIndexes
            res = theseModels[model].model.predict(nextData[:,theseIndexes])
            inputSet = np.dstack((inputSet, res))
            correct, percent = theseModels[model].model.computeScore(res, labels)
            print('correct', correct, 'fraction correct', percent)

        ts = inputSet.shape
        
        if len(ts) > 2 :
            inputSet = inputSet.reshape((ts[0],ts[1]*ts[2]))
        
        if level == numLevels-1 :
            return inputSet
        
        inputSet = transformSet[level].transform(inputSet)
        

        nextData = expand(inputSet, basis)
        print('nextData.shape',inputSet.shape)
    return inputSet