from os import path, listdir, environ
from os.path import join
environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # hide info logs, but not warnings, errors, or fatals
from keras_gen_and_image_arrays import *
from config import *

from PIL import Image as PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
import cv2 as cv


#Much of this code was taken from https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71


# Debug Setup
print('\n')
def debugPrintLeft(propertiesDict, leftAlignSpace=40):
    for propertyLabel, value in propertiesDict.items():
        print(f"{propertyLabel:<{leftAlignSpace}}{value}")
workingDir = path.dirname(path.realpath(__file__))
saveDir = path.dirname(saveLocation)
debugPrintLeft({
    "Pillow version:": PIL.__version__,
    "Tensorflow version:": tf.__version__,
    "Eager execution:": tf.executing_eagerly(),
    "Working directory:": workingDir,
    "Save path:": saveDir,
    "Loading from:": f"{workingDir}\\{saveLocation}",
    "Exposed GPUs:": tf.config.experimental.list_physical_devices('GPU'),
    "Exposed CPUs:": tf.config.experimental.list_physical_devices('CPU')
})



##Validate working directory and count files for each class
print("\n\nLooking through dataset directory ... ", end="")
workingDirContents = [f for f in listdir(workingDir) if path.isdir(join(workingDir, f))]
assert datasetName in workingDirContents and "Missing \'Dataset\' folder"
for folder in workingDirContents:
    print(f" - {folder}", end="")
    if folder == datasetName:
        print("*")
    else:
        print()



print(f"\n\nEncoding data labels from {datasetName} into numpy array...")
datasetDir = f"{workingDir}\\{datasetName}"
classFolderNames = [f for f in listdir(datasetDir) if path.isdir(join(datasetDir, f))]
numClasses = len(classFolderNames)
numClassFilesList = []
masterFileDirList = []
for classIndex, classFolder in enumerate(classFolderNames):
    classPath = f"{datasetDir}\\{classFolder}"
    oneHotIndices = []
    numFiles = 0
    for f in listdir(classPath):
        if path.isfile(join(classPath, f)):
            masterFileDirList.append(join(classPath, f))
            numFiles += 1
    numClassFilesList.append(numFiles)
    # NOTE: Max size of a python list on a 32 bit system is 536,870,912 elements
    print(f" - {classFolder} > {numFiles} files")
totalNumFiles = sum(numClassFilesList)
oneHotImageDirs = np.zeros((totalNumFiles, numClasses+1), dtype=object)    # one-encoded numpy array, but rightmost column is for filepaths
startArangeIndex = 0
for classIndex, numClassFiles in enumerate(numClassFilesList):
    oneHotImageDirs[np.arange(startArangeIndex, startArangeIndex+numClassFiles), classIndex] = 1
    startArangeIndex += numClassFiles
oneHotImageDirs[:, -1] = masterFileDirList
print("One-hot encoded labels array shape:", oneHotImageDirs.shape)
print(f"{totalNumFiles} TOTAL IMAGE FILES")



print("\n\nShuffling dataset...")
np.random.seed(randomSeed) #Setting a seed allows re-training the same model without contaminiating the test dataset
np.random.shuffle(oneHotImageDirs)
for i in range(min(totalNumFiles, 3)):
    print(f"{str(i)+':':<5}{oneHotImageDirs[i]}")

testAmount = int(splitPercentage*totalNumFiles)
print(f"\nSetting aside {testAmount} ({round(splitPercentage*100, 2)}%) images for testing dataset")
testDirs, trainDirs = oneHotImageDirs[:testAmount], oneHotImageDirs[testAmount:]
print(f"testDirs: {len(testDirs)}, trainDirs: {len(trainDirs)}")



#Creating Custom Generators
print("\n\nCreating custom keras generators...")
trainGen = KerasGeneratorFromFile(trainDirs, batchSize)
testGen = KerasGeneratorFromFile(testDirs, batchSize)
print("Test dataset generator: ", end="")
sampleBatch = trainGen.__getitem__(len(trainDirs) // batchSize)
print(f"({sampleBatch[0].shape}, {sampleBatch[1].shape}) shape returned")



print("\n\nGenerating model...")
model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), activation='relu',input_shape=(imgSize[0], imgSize[1], numColorChannels)),
    BatchNormalization(),
    Dropout(dropoutAmount),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    BatchNormalization(),
    Dropout(dropoutAmount),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    BatchNormalization(),
    Dropout(dropoutAmount),
    Flatten(),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(dropoutAmount)
])
for layer in internalLayers:    # Loop to add hidden Layers
    model.add(Dense(layer, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(dropoutAmount))
#Output Layer
model.add(Dense(numClasses, activation="softmax"))

#Finish the model
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

#Save initial
#model.save_weights(workingDir+"\\"+saveLocation.format(epoch=0))

model.summary()

#Prepare for saving
cp_callback = tf.keras.callbacks.ModelCheckpoint(workingDir=saveLocation,
                                                 save_weights_only=True,
                                                 verbose=1)


##Attempt to load data
if(saveLoad):
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
    #Training Time
    model.fit(x=trainGen,
        steps_per_epoch = int(len(trainDirs) // batchSize),
        epochs = numEpochs,
        verbose = sharedVerbose,
        validation_data = testGen,
        validation_steps = int(len(testDirs) // batchSize),
        callbacks=[cp_callback])
else:
    print("Training Skipped")
#Optional Model Export
if(exportWhenComplete):
    print("Exporting Model")
    model.save(exportLocation)

print("Starting Demo")
##This video camera demo tends to not work as your camera input isn't like the training data at all. It would be fun if it worked but it doesn't
'''
capture = cv.VideoCapture(camNum)
while True:
    isTrue,frame = capture.read()
    picture = prepareArrayImage(np.asarray(frame))
    if(isGrayScale):
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
'''
##This alternative demo usesing test files. Ensure your trained model is on the same shuffle seed to prevent contamination
print(int(len(testDirs)/batchSize))
for batchNum in range(int(len(testDirs)/batchSize)):
    batchImg,batchLabel = testGen.__getitem__(batchNum)
    predictions = model.predict(batchImg)
    for i in range(len(batchImg)):
        highestConfidence = max(predictions[i])
        highestIndex = predictions[i].tolist().index(highestConfidence)
        predictedCatagory = str(classFolderNames[batchLabel[i].tolist().index(1)]) + " : "
        if(highestConfidence < 0.5) :
            predictedCatagory += "None ("+str(classFolderNames[highestIndex]) +" "+ str(int(10000*highestConfidence)/100)+"%)"
        else:
            predictedCatagory += str(classFolderNames[highestIndex]) +" "+ str(int(10000*highestConfidence)/100)+"%"
        cv.imshow(predictedCatagory,cv.resize(np.uint8(batchImg[i]), (600, 600)))
        print("Showing "+str(i)+" predicted as "+str(predictedCatagory))
        cv.waitKey(0)##Wait indefinitly until a key is pressed5
    print(" -- batchNum: "+str(batchNum))
print("All Done")
