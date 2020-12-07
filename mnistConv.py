import numpy as np
import modelAdapter as mr 
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from polynomial import *
from stackedGeneralization import *
import ray
from layerData import *
import sys, traceback
import pickle

try :
    ## You need to define the ray initialization.  If you don't define the num_cpus it will use all available resources.
    ray.init(num_cpus=4)

    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    useTraining = 60000

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=useTraining, test_size=10000)

    #Put in the range -1 to 1
    X_train = np.array(X_train).astype(np.float16)/128.0 - 1.0
    y_train = np.array(y_train).astype(np.int)

    X_test = np.array(X_test).astype(np.float16)/128 - 1.0
    y_test = np.array(y_test).astype(np.int)

    ## The first layer is expanded differently than the internal
    ## layers since the user may know something about the input data
    firstExpansion = polynomial.basis0
    data = expand(X_train, firstExpansion)
    dataTest = expand(X_test, firstExpansion)

    #Info for creating the model
    layerDetails = []

    rfPrototype = mr.RandomForest(10,maxDepth=20, n_jobs=4, n_estimators=100)
    rfPrototypeFinal = mr.RandomForest(10, maxDepth=20, n_jobs=4, n_estimators=1000)

    lsPrototype = mr.PolynomialClassification(nLabels=10)

    tPrototype = lsPrototype

    convPrototype = mr.Convolutional2D(tPrototype, 28, 28, 4, 4, 4, maxSamples = 100000)
    convPrototype2 = mr.Convolutional2D(tPrototype, 7, 7, 4, 1, 1, maxSamples = 100000)
    

    '''
    convPrototype = mr.Convolutional2D(tPrototype, 28, 28, 3, 1, 1, maxSamples = 100000)
    convPrototype2 = mr.Convolutional2D(tPrototype, 26, 26, 3, 1, 1, maxSamples = 100000)
    convPrototype3 = mr.Convolutional2D(tPrototype, 24, 24, 2, 1, 2, maxSamples = 100000)
    convPrototype4 = mr.Convolutional2D(tPrototype, 12, 12, 3, 1, 1, maxSamples = 100000)
    convPrototype5 = mr.Convolutional2D(tPrototype, 10, 10, 3, 1, 1, maxSamples = 100000)
    convPrototype6 = mr.Convolutional2D(tPrototype, 8, 8, 2, 1, 2, maxSamples = 100000)
    '''
    
    '''
    convPrototype = mr.Convolutional2D(tPrototype, 28, 28, 4, 1, 4, maxSamples = 100000)
    convPrototype2 = mr.Convolutional2D(tPrototype, 7, 7, 4, 1, 1, maxSamples = 100000)
    convPrototype3 = mr.Convolutional2D(tPrototype, 12, 12, 5, 1, 1, maxSamples = 100000)
    convPrototype4 = mr.Convolutional2D(tPrototype, 8, 8, 2, 2, 1, maxSamples = 100000)
    convPrototype5 = mr.Convolutional2D(tPrototype, 4, 4, 2, 1, 1, maxSamples = 100000)
    convPrototype6 = mr.Convolutional2D(tPrototype, 8, 8, 5, 1, 1, maxSamples = 100000)
    convPrototype7 = mr.Convolutional2D(tPrototype, 4, 4, 2, 1, 1, maxSamples = 100000)
    '''

    #Input layer
    layerDetails.append(LayerInfo(inputFeatures = FeatureMap.all, numberOfBaseModels = 1, maxSubModels = 10, learnerPrototype=convPrototype, expansionFunction=basis0))
    layerDetails.append(LayerInfo(inputFeatures = FeatureMap.all, numberOfBaseModels = 1, maxSubModels = 10, learnerPrototype=convPrototype2, expansionFunction=basis0))
    #layerDetails.append(LayerInfo(inputFeatures = FeatureMap.all, numberOfBaseModels = 1, maxSubModels = 1, learnerPrototype=convPrototype3, expansionFunction=basis0))
    #layerDetails.append(LayerInfo(inputFeatures = FeatureMap.all, numberOfBaseModels = 1, maxSubModels = 5, learnerPrototype=convPrototype4, expansionFunction=basis0))
    #layerDetails.append(LayerInfo(inputFeatures = FeatureMap.all, numberOfBaseModels = 1, maxSubModels = 5, learnerPrototype=convPrototype5, expansionFunction=basis0))
    #layerDetails.append(LayerInfo(inputFeatures = FeatureMap.all, numberOfBaseModels = 1, maxSubModels = 5, learnerPrototype=convPrototype6, expansionFunction=basis0))
    
    #layerDetails.append(LayerInfo(inputFeatures = FeatureMap.all, numberOfBaseModels = 1, maxSubModels = 1, learnerPrototype=convPrototype7, expansionFunction=basis0))

    #Hidden layer
    #layerDetails.append(LayerInfo(inputFeatures = FeatureMap.even, numberOfBaseModels = 20, maxSubModels = 7, expansionFunction=basis2))
    #layerDetails.append(LayerInfo(inputFeatures = FeatureMap.even, numberOfBaseModels = 10, maxSubModels = 5,learnerPrototype=rfPrototype, expansionFunction=basis0))

    #Output layer
    layerDetails.append(LayerInfo(inputFeatures = FeatureMap.all, numberOfBaseModels = 1, maxSubModels = 7, learnerPrototype=lsPrototype, expansionFunction=basis0))
    layerDetails.append(LayerInfo(inputFeatures = FeatureMap.all, numberOfBaseModels = 1, maxSubModels = 7, learnerPrototype=lsPrototype, expansionFunction=basis0))


    ## Build the stacked model - use almost all examples to build each model - can't use them all or every model will be identical.
    #allModelSets, transformSet, basis = buildParallel(data, y_train, layerDetails, int(0.95*useTraining), basis5, mr.MultiClassClassification(nLabels=10))
    allModelSets, transformSet, basis = buildParallel(data, y_train, layerDetails, int(0.95*useTraining), basis2DG)

    with open('mnistConv.model','wb') as mnistFile :
        pickle.dump([allModelSets, transformSet, basis], mnistFile)

    ## Now run through the test data
    print('testing ---------------------------------------------')
    evaluateModel(dataTest, y_test, allModelSets, transformSet, basis)
except :

    e = sys.exc_info()
    print(e[0])
    print(e[1])
    traceback.print_tb(e[2])

print('shutting down ray')
ray.shutdown()