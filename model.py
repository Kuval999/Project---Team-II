import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu

letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y'
        ]

def readTrainingData(trainingDirectory):
    imageData = []
    targetData = []
    for eachLetter in letters:
        if(eachLetter == '7' or eachLetter == '0'):
            num = 14
        elif (eachLetter == 'K' or eachLetter == '8' or eachLetter == '5' or eachLetter == '4' or eachLetter == '1'):
            num = 15
        elif (eachLetter == 'Y' or eachLetter == 'X' or eachLetter == 'T' or eachLetter == 'N' or eachLetter == 'G' or eachLetter == '6'):
            num = 11
        elif( eachLetter == 'L'):
            num = 12
        elif(eachLetter == 'J' or eachLetter == '3'):
            num = 13
        elif(eachLetter == 'A' or eachLetter == '2'):
            num = 18
        else:
            num = 10
        for each in range(num):
            image_path = os.path.join(trainingDirectory, eachLetter, eachLetter + '_' + str(each) + '.jpg')
            # read each image of each character
            img_details = imread(image_path, as_grey=True)
            # converts each character image to binary image
            binaryImage = img_details < threshold_otsu(img_details)
            binaryImage.resize(20,20)
            # the 2D array of each image is flattened because the machine learning
            # classifier requires that each sample is a 1D array
            # therefore the 20*20 image becomes 1*400
            # in machine learning terms that's 400 features with each pixel
            # representing a feature
            flatBinImage = binaryImage.reshape(-1)
            imageData.append(flatBinImage)
            targetData.append(eachLetter)            

    return (np.array(imageData), np.array(targetData))

def crossValidation(model, numOfFold, trainData, trainLabel):
    # this uses the concept of cross validation to measure the accuracy
    # of a model, the num_of_fold determines the type of validation
    # e.g if num_of_fold is 4, then we are performing a 4-fold cross validation
    # it will divide the dataset into 4 and use 1/4 of it for testing
    # and the remaining 3/4 for the training
    accuracyResult = cross_val_score(model, trainData, trainLabel,
                                      cv=numOfFold)
    print("Cross Validation Result for ", str(numOfFold), " -fold")

    print(accuracyResult * 100)


currentDir = os.path.dirname(os.path.realpath(__file__))

trainingDatasetDir = os.path.join(currentDir, 'train')

imageData, targetData = readTrainingData(trainingDatasetDir)

# the kernel can be 'linear', 'poly' or 'rbf'
# the probability was set to True so as to show
# how sure the model is of it's prediction
svcModel = SVC(kernel='linear', probability=True)

crossValidation(svcModel, 4, imageData, targetData)

# let's train the model with all the input data
svcModel.fit(imageData, targetData)

# we will use the joblib module to persist the model
# into files. This means that the next time we need to
# predict, we don't need to train the model again
saveDirectory = os.path.join(currentDir, 'models/svc/')
if not os.path.exists(saveDirectory):
    os.makedirs(saveDirectory)
joblib.dump(svcModel, saveDirectory+'/svc.pkl')
