import numpy as np
import multiclassRegression as mr 
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import polynomial
from polynomialRegression import *
import ray

## You need to define the ray initialization.  If you don't define the num_cpus it will use all available resources.
ray.init(num_cpus=1)

X, y = fetch_openml('diabetes', version=1, return_X_y=True)

#Use 90% for training and 10% for test
useTraining = int(0.9*X.shape[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=useTraining, test_size=X.shape[0]-useTraining)

le = LabelEncoder()
le.fit(['tested_positive','tested_negative'])
y_train = le.transform(y_train)
y_test = le.transform(y_test)

scaler = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

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

## Build the stacked model - each model is built on a random subset of 95% of the training data
allModelSets, transformSet, basis = buildParallel(data, y_train, modelsInLayer, int(0.95*useTraining), polynomial.basis5, mr.MultiClassClassification(nLabels=2))

## Now run through the test data
print('testing ---------------------------------------------')
evaluateModel(dataTest, y_test, allModelSets, transformSet, basis)