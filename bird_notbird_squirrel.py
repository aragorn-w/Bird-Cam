import random
from math import ceil
from os import path, listdir, environ
from os.path import join
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from PIL import Image as PIL
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization
# hide info logs=1, but not warnings=2, errors=3, or fatals=4
# environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


# Much of this code was taken from https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
# ConfigVars
datasetName = "Dataset"
saveLocation = "OldSavedModels\\Checkpoints\\cp-{epoch:04d}.ckpt"
exportLocation = "OldSavedModels\\ExportedModels\\model_export"
saveLoad = True    # True means load recent save before training if it exists
exportWhenComplete = True
randomSeed = 42069
batchSize = 64
imgSize = (224, 224)
numEpochs = 1  # 0 for no training(Usefull for exporting)
sharedVerbose = 1
colorCode = 'RGB'
numColorChannels = 3    # change to 1 when colorCode is 'L', grayscale
splitPercentage = 0.05
internalLayers = [32, 32]
dropoutAmount = 0.25  # For preventing overfitting. 0.25 is plenty, 0 to disable
camNum = 0


# Functions & Classes
class KerasGeneratorFromFile(keras.utils.Sequence):
    def __init__(self, fileArray, batch_size):
        self.fileArray = fileArray
        self.batch_size = batch_size

    def __len__(self):
        return int(ceil(float(len(self.fileArray)) / float(self.batch_size)))

    def __getitem__(self, idx):
        # assert (((idx+1) * self.batch_size) <= len(self.fileArray))
        data = []
        labels = []
        for fileID in range((idx * self.batch_size), ((idx+1) * self.batch_size)-1):
            if(fileID >= len(self.fileArray)):
                break
            data.append(prepareFileImage(self.fileArray[fileID][0]))
            labels.append(self.fileArray[fileID][1])    # This is run every time a batch is requested, it is run a bunch and it isn't the fastest. Too Bad!
        return (np.reshape(np.array(data), (len(data), imgSize[0], imgSize[1], numColorChannels)), np.array(labels))


def prepareImage(img):
    return np.array(img.convert(colorCode).resize(imgSize))


def prepareArrayImage(array):
    return prepareImage(PIL.fromarray(array))


def prepareFileImage(imgPath):
    return prepareImage(PIL.open(imgPath))


# Debug Setup
print('\n\n')
print('Pillow Version:', PIL.__version__)
print("Tensorflow is running at:", tf.__version__)
print("Eager execution:", tf.executing_eagerly())
filePath = (path.dirname(path.realpath(__file__)))
print("Applicaiton running in "+filePath)
saveDir = path.dirname(saveLocation)
print("Saving to "+saveDir)
print(f"Loading From {filePath}\\{saveLocation}")
print("Exposed GPU's " + str(tf.config.experimental.list_physical_devices('GPU')))
print("Exposed CPU's " + str(tf.config.experimental.list_physical_devices('CPU')))
print()
print("Looking for dataset directory:", end="")

# Get files
directories = [f for f in listdir(filePath) if path.isdir(join(filePath, f))]
assert len(directories) > 0
print("Found "+str(len(directories))+" directories")
assert datasetName in directories
for directory in directories:
    print(" - "+str(directory), end="")
    if(str(directory) == datasetName):
        print("*")
    else:
        print()
print()
print("Getting Classifiers "+str(datasetName))
directories = [f for f in listdir(f"{filePath}\\{datasetName}") if path.isdir(join(filePath+"\\"+datasetName, f))]
for directory in directories:
    print(" - "+str(directory) + " > "+str(len([f for f in listdir(f"{filePath}\\{datasetName}\\{str(directory)}") if path.isfile(join(filePath+"\\"+datasetName+"\\"+str(directory), f))])) + "# files")
print()

# Actual Doing stuff
print("Turning Data into an array")
# You win this time, Bryce. But your precious lists won't last...
# Not against my numpy arrays muahahaha

masterFileArray = []
for i in range(len(directories)):
    files = [f for f in listdir(f'{filePath}\\{datasetName}\\{str(directories[i])}') if path.isfile(join(filePath+"\\"+datasetName+"\\"+str(directories[i]), f))]
    for file in files:
        oneHot = [0] * len(directories)
        oneHot[i] = 1
        masterFileArray.append((filePath+"\\"+datasetName+"\\"+str(directories[i])+"\\"+str(file), oneHot))

print("Shuffling.")
random.seed(randomSeed)  # Good for repeating conditions
random.shuffle(masterFileArray)

print(str(len(masterFileArray)) + "# Array Elements")
for i in range(min(len(masterFileArray), 10)):
    print(str(i) + " : " + str(masterFileArray[i]))


testAmount = int(splitPercentage*len(masterFileArray))
print(f"Setting aside {testAmount} as testing ({round(splitPercentage*100, 2)} of total)")
testFiles = masterFileArray[:testAmount]
trainFiles = masterFileArray[testAmount:]
print("testFiles : "+str(len(testFiles)))
print("trainFiles : "+str(len(trainFiles)))
print()

# Creating Custom Generators
print("Creating Generators")
trainGen = KerasGeneratorFromFile(trainFiles, batchSize)
testGen = KerasGeneratorFromFile(testFiles, batchSize)
print("Testing Generator : ", end="")
sampleImages = trainGen.__getitem__(int(len(trainFiles) // batchSize))
print("("+str(sampleImages[0].shape)+","+str(sampleImages[1].shape)+") shape returned")


model = Sequential()
model.add(Conv2D(input_shape=(imgSize[0], imgSize[1], numColorChannels), filters=16, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(dropoutAmount))
model.add(Flatten())

# Start of regular Network
for layer in internalLayers:  # Hidden Layers
    model.add(Dense(layer, activation="relu"))
    model.add(Dropout(dropoutAmount))
# Output Layer
model.add(Dense(len(directories), activation="softmax"))


# Finish the model
# NOTE: loss hyperparameter might have to be changed to binary crossentropy for binary classification
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# Save initial
# model.save_weights(filePath+"\\"+saveLocation.format(epoch=0))

model.summary()

# Prepare for saving
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=saveLocation,
    save_weights_only=True,
    verbose=1
)


# Attempt to load data
if saveLoad:
    print("Attempting to load previous weights")
    try:
        latest = tf.train.latest_checkpoint(saveDir)
        model.load_weights(latest)
        loss, acc = model.evaluate(testGen, verbose=sharedVerbose)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    except:
        print("Error in loading: Skipping Loading")
        saveLoad = False
if(not(saveLoad)):
    model.save_weights(saveLocation.format(epoch=0))

print()
if(numEpochs != 0):
    print("Training Start")
    # Training Time
    model.fit(
        x=trainGen,
        steps_per_epoch=int(len(trainFiles) // batchSize),
        epochs=numEpochs,
        verbose=sharedVerbose,
        validation_data=testGen,
        validation_steps=int(len(testFiles) // batchSize),
        callbacks=[cp_callback])
else:
    print("Training Skipped")
# Optional Model Export
if(exportWhenComplete):
    print("Exporting Model")
    model.save(exportLocation)


# This video camera demo tends to not work,
# as your camera input isn't like the training data at all.
# It would be fun if it worked, but it doesn't.
'''
print("Starting Demo")
capture = cv.VideoCapture(camNum)
while True:
    isTrue,frame = capture.read()
    picture = prepareArrayImage(np.asarray(frame))
    if colorCode == 'L' and numColorChannels == 1:  # if grayscale
        picture = np.reshape(np.array(picture),(imgSize[0],imgSize[1],1))
    else:
        picture = np.reshape(np.array(picture),(imgSize[0],imgSize[1],3))
    cv.imshow('frame',cv.resize(np.uint8(np.array(picture)), (500, 500)))#1000 Output Size
    picture=np.array([picture])
    predictions = model.predict(picture)
    predictions = predictions.tolist()
    highestConfidence = max(predictions[0])#Predictions a 1 element array don't question it
    highestIndex = predictions[0].index(highestConfidence)
    if(highestConfidence < 0.5) :
        print("None (",end="")
        print(str(directories[highestIndex]) +" "+ str(int(1000*highestConfidence)/10)+"%)")
    else:
        print(str(directories[highestIndex]) +" "+ str(int(1000*highestConfidence)/10)+"%")


    if cv.waitKey(20) & 0xFF==ord('d'):
        break


# This alternative demo usesing test files.
# Ensure your trained model is on the same
# shuffle seed to prevent contamination
for batchNum in range(int(len(testFiles)/batchSize)):
    batchImg, batchLabel = testGen.__getitem__(batchNum)
    predictions = model.predict(batchImg)
    for i in range(len(batchImg)):
        highestConfidence = max(predictions[i])
        highestIndex = predictions[i].tolist().index(highestConfidence)
        predictedCatagory = str(directories[batchLabel[i].tolist().index(1)]) + " : "
        if(highestConfidence < 0.5):
            predictedCatagory += "None ("+str(directories[highestIndex]) +" "+ str(int(10000*highestConfidence)/100)+"%)"
        else:
            predictedCatagory += str(directories[highestIndex]) +" "+ str(int(10000*highestConfidence)/100)+"%"
        cv.imshow(predictedCatagory, cv.resize(np.uint8(batchImg[i]), (600, 600)))
        print("Showing "+str(i)+" predicted as "+str(predictedCatagory))
        cv.waitKey(0)  #Wait indefinitly until a key is pressed5
    print(" -- batchNum: "+str(batchNum))
print("All Done")
'''
