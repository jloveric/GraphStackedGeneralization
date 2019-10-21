import unittest
from polynomialRegression import *
from layerData import *
from polynomial import *
import numpy as np
import cv2
import os

class TestHSpace(unittest.TestCase):
    
    def test_constructLayerInputs(self):
        
        layerDetails = []

        #Input layer
        layerDetails.append(LayerInfo(inputFeatures = FeatureMap.even, numberOfBaseModels = 10, maxSubModels = 5, expansionFunction=basis0))
        data = np.arange(0,10000).reshape((200,50))

        ans = np.array([[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24],
                [25, 26, 27, 28, 29],
                [30, 31, 32, 33, 34],
                [35, 36, 37, 38, 39],
                [40, 41, 42, 43, 44],
                [45, 46, 47, 48, 49]])

        constructLayerInputs(layerDetails[0], data)
        self.assertTrue(np.array_equal(layerDetails[0].inputIndexes,ans))
        print('layerDetails',layerDetails[0].inputIndexes)

        #Check with all connectivity
        layerDetails.append(LayerInfo(inputFeatures = FeatureMap.all, numberOfBaseModels = 10, maxSubModels = 5, expansionFunction=basis0))
        constructLayerInputs(layerDetails[1], data)
        self.assertTrue(layerDetails[1].inputIndexes.shape ==(10,50))
        print('layerDetails[1][0]',layerDetails[1].inputIndexes[0])
        print(layerDetails[1].inputIndexes.shape)

        #Check with uneven division
        layerDetails.append(LayerInfo(inputFeatures = FeatureMap.even, numberOfBaseModels = 11, maxSubModels = 5, expansionFunction=basis0))
        constructLayerInputs(layerDetails[2], data)
        self.assertTrue(layerDetails[2].inputIndexes[10].size==10)
        self.assertTrue(layerDetails[2].inputIndexes[0].size==4)
        print(layerDetails[2].inputIndexes)


    
if __name__ == '__main__':
    unittest.main()