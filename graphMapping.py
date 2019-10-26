'''
Hierarchical eigenspace in python
https://www.flickr.com/photos/cascadebicycleclub/47150120251/in/album-72157706807506035/
'''

from scipy import misc
import numpy as np
import logging

#Compute the training samples for the original data for a convolutional layer. Of course
#these aren't really convolutions, they are just locally receptive fields.
def createTrainingSamples2Dfrom1D(width, height, sampleWidth, stride, maxSamples, data, labels) :

    examples = data.shape[0]
    
    size = data.shape[1]

    channels = int(size/(width*height))
    logging.debug('width', width, 'height', height,'data.shape', data.shape)
    dataNew = data[:examples].reshape((examples,width,height,channels))
    
    newSet = []
    newLabels = []

    count = 1
    case = 0
    while case < examples and count < maxSamples :

        label = labels[case]

        for i in range(0, height-sampleWidth+1, stride) :
            for j in range(0, width-sampleWidth+1, stride) :
                newSet.append(dataNew[case,i:i+sampleWidth,j:j+sampleWidth,:].flatten())
                newLabels.append(label)
                count = count + 1
        
        case = case + 1

                    
    return np.array(newSet), np.array(newLabels)

#Just create the indexes for each of the models in a convolutional layer.  Yes each model is
#actually identical so we don't need to repeat them.
def createInput2DMapping(width, height, stride, sampleWidth) :

    indexes = np.arange(0,width*height).reshape((height,width))
    indexSet = []
    
    for i in range(0, height-sampleWidth+1, stride) :
        for j in range(0, width-sampleWidth+1, stride) :
            indexSet.append(indexes[i:i+sampleWidth,j:j+sampleWidth].flatten())      
                    
    return np.array(indexSet)

def applyModel(input, inputMapping, model) :

    #print('input.shape', input.shape, 'inputMapping.shape', inputMapping.shape)

    output = []
    for i in range(0, len(inputMapping)) :
        ni = input[:,inputMapping[i]]
        output.append(model.predict(input[:,inputMapping[i]]))
    
    final = np.array(output).transpose((1,0,2))
    shape = final.shape 
    final = final.reshape((shape[0],shape[1]*shape[2]))
    #print('final.shape applyModel', final.shape)
    return final