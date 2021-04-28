import numpy as np
from tensorflow import keras
from math import ceil
from PIL import Image

from configvars import *


# Functions & Classes
class KerasGeneratorFromFile(keras.utils.Sequence):
    def __init__(self, fileArray, batchSize: int) :
        self.fileArray = fileArray
        self.batchSize = batchSize
        if numColorChannels == 1:
            self.colorType = 'L'
        elif numColorChannels == 3:
            self.colorType = 'RGB'

    def __len__(self):
        return int(ceil(float(len(self.fileArray)) / float(self.batchSize)))   # get the number of batches

    def __getitem__(self, batchIndex):
        data = np.empty((self.batchSize, imgSize[0], imgSize[1], numColorChannels))
        startIndex = batchIndex * self.batchSize
        for fileIndex in range(self.batchSize):
            fileIndex += startIndex
            assert fileIndex < len(self.fileArray)
            data[fileIndex] = np.asarray(Image.open(self.fileArray[fileIndex][-1]).convert(self.colorType).resize(imgSize))
        return (data, np.array(self.fileArray[:, :-1]).astype('float32'))