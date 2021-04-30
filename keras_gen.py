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
		startIndex = batchIndex * self.batchSize
		data = np.empty((len(self.fileArray)-startIndex, imgSize[0], imgSize[1], numColorChannels))
		for indexOffset in range(self.batchSize):
			fileIndex = startIndex + indexOffset
			if fileIndex >= len(self.fileArray):
				break
			# Convert to PIL image with wanted color channel, resize it, then translate it to a numpy array
			# print(self.fileArray[fileIndex])
			try:
				data[indexOffset] = preparePIL_Image(Image.open(self.fileArray[fileIndex][-1]))
			except:
				print('\n\nThe image URL:', self.fileArray[fileIndex][-1])
				print('The fileIndex:', fileIndex)
				print('The length of data:', len(data))
				print('The shape of data:', data.shape)
				input('HUHHH????')
			# print('DEBUGGING fileIndex:', fileIndex)
		return data, np.array(self.fileArray[startIndex:startIndex+self.batchSize, :-1]).astype('float32')


def prepareArrayImage(imageArray):
	# read into PIL, convert color channel, then resize
	return np.reshape(
		np.array(
			Image.fromarray(imageArray).convert(colorChannelType).resize(imgSize)
		),
		(imgSize[0], imgSize[1], numColorChannels)
	)


def preparePIL_Image(imgData):
	return np.array(imgData.convert(colorChannelType).resize(imgSize))
	# return np.reshape(
	# 	np.array(
	# 		imgData.convert(colorChannelType).resize(imgSize)
	# 	),
	# 	(imgSize[0], imgSize[1], numColorChannels)
	# )

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