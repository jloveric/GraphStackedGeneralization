import numpy as np
import enum

'''
Enumeration to show how to how the layer uses features from the previous
layer.  The standard would be all.  A convolutional network would be an example
that only uses some features per model per layer.
'''
class FeatureMap(enum.Enum) :
    all = 0
    even = 1
    overlap = 2
    random = 3 

class LayerInfo :
    def __init__(self, inputFeatures = FeatureMap.all) :
        self.inputs = inputs
        self.inputIndexes = None
        self.useAllInput = useAllInput
    
class ModelData :
    def __init__(self,inputIndexes = None, useAllInput = True, model = None) :
        self.inputIndexes = inputIndexes
        self.model = model

    def setModel(self, model) :
        self.model = model
