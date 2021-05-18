from os import path, listdir, environ
from os.path import join
environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # hide info logs=1, but not warnings=2, errors=3, or fatals=4
from keras_gen import *
from configvars import *

from PIL import Image as PIL
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
import cv2 as cv


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
print('totalNumFiles', totalNumFiles)
if numClasses > 2: #OneHot
    startArangeIndex = 0
    oneHotImageDirs = np.zeros((totalNumFiles, numClasses+1), dtype=object)    # one-encoded numpy array, but rightmost column is for filepaths
    for classIndex, numClassFiles in enumerate(numClassFilesList):
        oneHotImageDirs[np.arange(startArangeIndex, startArangeIndex+numClassFiles), classIndex] = 1.
        startArangeIndex += numClassFiles
elif numClasses == 2: #Binary
    oneHotImageDirs = np.zeros((totalNumFiles, 2), dtype=object) #Initialise
    oneHotImageDirs[:numClassFilesList[0], 0] = 1. #Set the first class to 1
else:
	assert False and "Invalid number of classes"
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
numTrainImages, numTestImages = len(trainDirs), len(testDirs)
numTrainBatches, numTestBatches = numTrainImages // batchSize, numTestImages // batchSize
print(f"# of test images: {numTestImages}, # of train images: {numTrainImages}")



#Creating Custom Generators
print("\n\nCreating custom keras generators...")
trainGen = KerasGeneratorFromFile(trainDirs, batchSize)
testGen = KerasGeneratorFromFile(testDirs, batchSize)
print("Test dataset generator: ", end="")
sampleBatch = trainGen.__getitem__(0)   # get first batch
print(f"({sampleBatch[0].shape}, {sampleBatch[1].shape}) shape returned")



print("\n\nGenerating model...")
model = Sequential([
    Conv2D(filters=8, kernel_size=(3,3), activation='relu',input_shape=(imgSize[0], imgSize[1], numColorChannels)),
    # BatchNormalization(),
    # Dropout(dropoutAmount),
    MaxPooling2D(pool_size=(2,2)),

    # Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    # BatchNormalization(),
    # Dropout(dropoutAmount),
    # MaxPooling2D(pool_size=(2,2)),

    Conv2D(filters=8, kernel_size=(3,3), activation='relu'),
    # BatchNormalization(),
    Dropout(dropoutAmount),
    Flatten(),

    Dense(16, activation="relu"),
    # BatchNormalization(),
    # Dropout(dropoutAmount)
])
for layerSize in denseLayerSizes:    # Loop to add hidden Layers
    model.add(Dense(layerSize, activation="relu"))
    # model.add(BatchNormalization())
    # model.add(Dropout(dropoutAmount))


# [print(i.shape, i.dtype) for i in model.inputs]
# [print(o.shape, o.dtype) for o in model.outputs]
# [print(l.name, l.input_shape, l.dtype) for l in model.layers]



#Compile, save initialization, and summarize the model
if (numClasses > 2):
    model.add(Dense(numClasses, activation="softmax"))  # Classification Layer
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy']) #Compile
else:
    model.add(Dense(1, activation = "sigmoid")) #Output Layer
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])#Compile

# model.save_weights(workingDir+"\\"+saveLocation.format(epoch=0))
print('\n\n')
model.summary()
#Prepare for saving
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=saveLocation, save_weights_only=True, verbose=1)



##Attempt to load data
# if saveLoad:
#     print("\n\nAttempting to load previous weights...")
#     try:
#         latest = tf.train.latest_checkpoint(saveDir)
#         model.load_weights(latest)
#         loss, acc = model.evaluate(testGen, verbose=sharedVerbose)
#         print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
#     except:
#         print("Error in loading: Skipping Loading")
#         saveLoad = False
# if not saveLoad:
#     model.save_weights(saveLocation.format(epoch=0))



if numEpochs != 0:
    print("\n\n Training model...")
    #Training Time
    model.fit(
        x=trainGen,
        steps_per_epoch = numTrainBatches-1,
        epochs = numEpochs,
        verbose = sharedVerbose,
        validation_data = testGen,
        validation_steps = numTestBatches-1,
        callbacks=[cp_callback])
else:
    print("\n\nTraining skipped")



#Optional Model Export
# if exportWhenComplete:
#     print("Exporting Model")
#     model.save(exportLocation)



print("Starting Demo")
##This video camera demo tends to not work as your camera input isn't like the training data at all. It would be fun if it worked but it doesn't
'''
capture = cv.VideoCapture(camNum)
while True:
    isTrue,frame = capture.read()
    picture = prepareArrayImage(np.asarray(frame))
    picture = np.reshape(np.array(picture),(imgSize[0],imgSize[1],numColorChannels))

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
print(numTestBatches)
for batchNum in range(numTestBatches):
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