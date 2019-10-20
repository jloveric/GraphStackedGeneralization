import numpy as np
import multiclassRegression as mr 
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import polynomial
from polynomialRegression import *
import ray
import matplotlib.pyplot as plt

## You need to define the ray initialization.  If you don't define the num_cpus it will use all available resources.
ray.init(num_cpus=1)

numVals = 10000
x = 2.0*np.arange(numVals)/(numVals-1)-1.0
x= x.reshape((x.shape[0],1))
print('x.shape', x.shape)
y = 0.5*np.sin(16.0*x)

#Use 90% for training and 10% for test
useTraining = int(0.9*numVals)

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=useTraining, test_size=x.shape[0]-useTraining)

## The first layer is expanded differently than the internal
## layers since the user may know something about the input data
basis = polynomial.basis2
firstExpansion = basis
data = expand(X_train, firstExpansion)
dataTest = expand(X_test, firstExpansion)

#Info for creating the model
width = 1
numLayers = 10
modelsInLayer = [width]*numLayers
modelsInLayer[-1] = 1

## Build the stacked model - each model is built on a random subset of 95% of the training data
allModelSets, transformSet, basis = buildParallel(data, y_train, modelsInLayer, int(1.0*useTraining), basis, mr.MultiOutputRegression(nOutputs=1, tolerance=0.1))

## Now run through the test data
print('testing ---------------------------------------------')
ans = evaluateModel(dataTest, y_test, allModelSets, transformSet, basis)
print('shape', X_test.shape, ans.shape)
plt.plot(X_test,ans,'x', x,y,'+')
plt.show()