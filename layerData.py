import numpy as np
import enum
from copy import deepcopy
from polynomial import *

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

'''
Information specific to how to construct a specific layer
'''
class LayerInfo :
    def __init__(self, inputFeatures = FeatureMap.all, numberOfBaseModels = 1, maxSubModels = 0, expansionFunction = basis0) :
        self.inputFeatures = inputFeatures
        self.inputIndexes = None
        self.numberOfBaseModels = numberOfBaseModels
        self.maxSubModels = maxSubModels
        self.expansionFunction = expansionFunction
    
    def clone(self) :
        return deepcopy(self)

'''
Information about how a specific model was constructed
'''    
class ModelData :
    def __init__(self,inputIndexes = None, useAllInput = True, model = None) :
        self.inputIndexes = inputIndexes
        self.model = model

    def setModel(self, model) :
        self.model = model

    def clone(self) :
        return deepcopy(self)
