'''
Hierarchical eigenspace in python
https://www.flickr.com/photos/cascadebicycleclub/47150120251/in/album-72157706807506035/
'''

from scipy import misc
import numpy as np
import logging

#Compute the training samples for the original data for a convolutional layer. Of course
#these aren't really convolutions, they are just locally receptive fields.
def createTrainingSamples2Dfrom1D(width, height, sampleWidth, stride, maxSamples, data, labels, dtype=np.float16) :

    examples = data.shape[0]
    
    size = data.shape[1]

    channels = int(size/(width*height))
    #print('width', width, 'height', height,'data.shape', data.shape, 'sampleWidth', sampleWidth,'stride', stride, 'maxSamples', maxSamples,'channels',channels)
    dataNew = data[:examples].reshape((examples,width,height,channels)).astype(dtype)
    #print('dataNew.shape', dataNew.shape)
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

                    
    return np.array(newSet,dtype=dtype), np.array(newLabels,dtype=dtype)

#Just create the indexes for each of the models in a convolutional layer.  Yes each model is
#actually identical so we don't need to repeat them.
def createInput2DMapping(width, height, stride, sampleWidth, channels) :
    #print('width', width, 'height', height, 'channels', channels)
    indexes = np.arange(0,width*height*channels).reshape((height,width,channels))
    indexSet = []
    
    for i in range(0, height-sampleWidth+1, stride) :
        for j in range(0, width-sampleWidth+1, stride) :
            indexSet.append(indexes[i:i+sampleWidth,j:j+sampleWidth,:].flatten())      
                    
    return np.array(indexSet,dtype=np.int32)

def applyModel(input, inputMapping, model, dtype=np.float16) :

    #print('input.shape', input.shape, 'inputMapping.shape', inputMapping.shape)
    output = []
    for i in range(0, len(inputMapping)) :
        ni = input[:,inputMapping[i]]
        output.append(model.predict(input[:,inputMapping[i]]))
    
    final = np.array(output,dtype=dtype).transpose((1,0,2))
    shape = final.shape 
    final = final.reshape((shape[0],shape[1]*shape[2]))
    #print('final.shape applyModel', final.shape)
    return final

'''
def maxPooling(input, inputMapping, model, dtype=np.float16) :
    #print('input.shape', input.shape, 'inputMapping.shape', inputMapping.shape)
    output = []
    for i in range(0, len(inputMapping)) :
        ni = input[:,inputMapping[i]]
        output.append(model.predict(input[:,inputMapping[i]]))
    
    final = np.array(output,dtype=dtype).transpose((1,0,2))
    shape = final.shape 
    final = final.reshape((shape[0],shape[1]*shape[2]))
    #print('final.shape applyModel', final.shape)
'''