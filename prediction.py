import os
import segmentation
from sklearn.externals import joblib

currentDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(currentDir, 'models/svc/svc.pkl')
model = joblib.load(modelDir)
count = 0
classificationResult = []
for eachCharacter in segmentation.characters:
    # converts it to a 1D array
    eachCharacter = eachCharacter.reshape(1, -1);
    result = model.predict(eachCharacter)
    classificationResult.append(result)

#print(classificationResult)

plateString = ''
for eachPredict in classificationResult:
    plateString += eachPredict[0]

#print(plateString)

# it's possible the characters are wrongly arranged
# since that's a possibility, the column_list will be
# used to sort the letters in the right order

columnListCopy = segmentation.columnList[:]
segmentation.columnList.sort()
rightplateString = ''
for each in segmentation.columnList:
    rightplateString += plateString[columnListCopy.index(each)]

print('\t\tOutput : ', rightplateString)