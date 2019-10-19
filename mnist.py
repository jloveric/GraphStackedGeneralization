import numpy as np
import multiclassRegression as mr 
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import polynomial
from polynomialRegression import *
import ray

## You need to define the ray initialization.  If you don't define the num_cpus it will use all available resources.
ray.init(num_cpus=4)

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

useTraining = 6000

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=useTraining, test_size=10000)

#Put in the range -1 to 1
X_train = np.array(X_train)/128.0 - 1.0
y_train = np.array(y_train).astype(int)

X_test = np.array(X_test)/128 - 1.0
y_test = np.array(y_test).astype(int)

## The first layer is expanded differently than the internal
## layers since the user may know something about the input data
firstExpansion = polynomial.basis0
data = expand(X_train, firstExpansion)
dataTest = expand(X_test, firstExpansion)

#Info for creating the model
width = 1
numLayers = 3
modelsInLayer = [width]*numLayers
modelsInLayer[-1] = 1

## Build the stacked model - use almost all examples to build each model - can't use them all or every model will be identical.
allModelSets, transformSet, basis = buildParallel(data, y_train, modelsInLayer, int(0.95*useTraining), polynomial.basis5, mr.MultiClassRegression(nLabels=10))

## Now run through the test data
print('testing ---------------------------------------------')
evaluateModel(dataTest, y_test, allModelSets, transformSet, basis)