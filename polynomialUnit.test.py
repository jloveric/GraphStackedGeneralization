import unittest
from polynomial import *
import numpy as np
import cv2
import os

class TestHSpace(unittest.TestCase):
    
    def test_createSamples2D(self):
        data = np.array([[-1, 2, 3, 4]])

        ans = basis5DG(data)
        print('ans', ans)
    
if __name__ == '__main__':
    unittest.main()