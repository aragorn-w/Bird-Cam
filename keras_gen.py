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

	def __len__(self):
		return int(ceil(float(len(self.fileArray)) / float(self.batchSize)))   # get the number of batches

	def __getitem__(self, batchIndex):
		data = np.empty((self.batchSize, imgSize[0], imgSize[1], numColorChannels))
		startIndex = batchIndex * self.batchSize
		for fileIndex in range(self.batchSize):
			fileIndex += startIndex
			assert fileIndex < len(self.fileArray)
			# Convert to PIL image with wanted color channel, resize it, then translate it to a numpy array
			data[fileIndex] = prepareArrayImage(Image.open(self.fileArray[fileIndex][-1]))
		return (data, np.array(self.fileArray[:, :-1]).astype('float32'))


def prepareArrayImage(imageArray):
	# read into PIL, convert color channel, then resize
	return np.array(Image.fromarray(imageArray).convert(colorChannelType).resize(imgSize))


'''
def prepareArrayImage(array):
    return prepareImage(PIL.fromarray(array))
def prepareImage(img):
    if(isGrayScale):
        return np.array(img.convert('L').resize(imgSize))
    else:
        return np.array(img.convert('RGB').resize(imgSize))

def prepareFileImage(imgPath):
    return prepareImage(PIL.open(imgPath))
'''