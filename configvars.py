# ConfigVars
datasetName = "Dataset"
saveLocation = "SavedModels\\Checkpoints\\cp-{epoch:04d}.ckpt"
exportLocation = "SavedModels\\ExportedModels\\model_export"
saveLoad = True             # Look for a recent save. If it exists: Load it before training
exportWhenComplete = False
randomSeed = 42069
batchSize = 1
imgSize = (224,224) 		# 32 v 32 reccomended
numEpochs = 100 				# 0 for no training (Useful for exporting)
sharedVerbose = 1
numColorChannels = 3
splitPercentage = 0.1       # 1.0 = 100%
internalLayers = [64,64]
dropoutAmount = 0.5 		# Used to prevent overfitting. 0.25 is more than plenty, 0 to disable
camNum = 0

if numColorChannels == 3:
	colorChannelType = 'RGB'
else:
	colorChannelType = 'L'