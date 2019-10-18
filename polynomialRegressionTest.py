import numpy as np
from mnist import MNIST
import multiclassRegression as mr 
from sklearn import preprocessing
import polynomial
from polynomialRegression import *
import ray

ray.init(num_cpus=4)

## get and prepare the data
mndata = MNIST('./python-mnist/data')
images, labels = mndata.load_training()
imagesTest, labelsTest = mndata.load_testing()

useTraining = 60000

npImages = np.array(images)
npLabels = np.array(labels)
npImages=npImages[:useTraining]
npLabels=npLabels[:useTraining]

## The input data is expected to be normalized between -1 and 1
npImages = (npImages/128)-1.0

npImagesTest = np.array(imagesTest)
npLabelsTest = np.array(labelsTest)

npImagesTest = (npImagesTest/128)-1.0

## The first layer is expanded differently than the internal
## layers since the user may know something about the input data
firstExpansion = polynomial.basis0
data = expand(npImages, firstExpansion)
dataTest = expand(npImagesTest, firstExpansion)

#Info for creating the model
width = 10
numLayers = 3
modelsInLayer = [width]*numLayers
modelsInLayer[-1] = 1

## Build the stacked model
#allModelSets, transformSet, basis = buildParallel(data, npLabels, modelsInLayer, int(2*useTraining/width), polynomial.basis2DG)
allModelSets, transformSet, basis = buildParallel(data, npLabels, modelsInLayer, int(0.95*useTraining), polynomial.basis5)

## Now run through the test data
print('testing ---------------------------------------------')
evaluateModel(dataTest, npLabelsTest, allModelSets, transformSet, basis)