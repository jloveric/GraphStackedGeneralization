'''
Hierarchical eigenspace in python
https://www.flickr.com/photos/cascadebicycleclub/47150120251/in/album-72157706807506035/
'''

from scipy import misc
import numpy as np

#Compute the training samples for the original data for a convolutional layer. Of course
#these aren't really convolutions, they are just locally receptive fields.
def createTrainingSamples2Dfrom1D(width, height, sampleWidth, stride, data, labels) :

    examples = data.shape[0]
    size = data.shape[1]
    channels = int(size/(width*height))

    dataNew = data.reshape((examples,width,height,channels))
    
    newSet = []
    newlabels = []
    for case in range(0,examples) :

        label = labels[case]

        for i in range(0, height-sampleWidth+1, stride) :
            for j in range(0, width-sampleWidth+1, stride) :
                newSet.append(dataNew[case,i:i+sampleWidth,j:j+sampleWidth,:].flatten())
                newLabels.append(label)
                    
    return newSet, newLabels

#Just create the indexes for each of the models in a convolutional layer.  Yes each model is
#actually identical so we don't need to repeat them.
def createInput2DMapping(width, height, stride) :

    indexes = np.arange(0,width*height).reshape((height,width))
    indexSet = []
    
    for i in range(0, height-sampleWidth+1, stride) :
        for j in range(0, width-sampleWidth+1, stride) :
            indexSet.append(indexes[i:i+sampleWidth,j:j+sampleWidth,:].flatten())      
                    
    return np.array(indexSet)

def applyModel(input, inputMapping, model) :

    output = []
    for i in range(0, len(inputMapping)) :
        output.append(model.classify(input[inputMapping[i]]))
    
    return np.array(output)