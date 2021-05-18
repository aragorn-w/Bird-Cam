import os

APP_FOLDER = 'C:\\Users\\wanga\\Desktop\\Personal\\Machine Learning\\Sophomore AI Class\\Bird, Not Bird Data\\not_bird'

totalFiles = 0
totalDir = 0
subDirs = []
firstRunDone = False

dirIndex = 0
for _, dirs, files in os.walk(APP_FOLDER):
    if not firstRunDone:
        subDirs = dirs
        firstRunDone = True
        continue
    totalDir += 1
    filesInDir = 0
    for Files in files:
        filesInDir += 1
    print(f'# of files in {subDirs[dirIndex]}: {filesInDir}')
    totalFiles += filesInDir
    dirIndex += 1


print('Total # of files',totalFiles)
print('Total # of directories',totalDir)
