# -*- coding: utf-8 -*-
"""
Character Segmentation on number plate
Steps - 
1. Invert the number plate image to get our region of interest on foreground
2. Use connected component analysis(CCA) - identifies and groups connected regions on the foreground
   Note - Connected region -> adjacent pixels have same value 
3. Draw rectangular boxes over connected regions
4. Apply filter to detect characters on number plate
    a. Assumption - Ratio of width and length of characters to license plate 
                  - Number plate area
"""

import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import detectLicensePlate
#Input to this module is the detected number plate from detectLicensePlate.py module

# The invert was done so as to convert the black pixel to white pixel and vice versa
licensePlate = np.invert(detectLicensePlate.plateLikeObjects[0])
#license_plate  = detectLicensePlate.plateLikeObjects[0]
labelledPlate = measure.label(licensePlate)

fig, ax1 = plt.subplots(1)
ax1.imshow(licensePlate, cmap="gray")
for regions in regionprops(labelledPlate):
    minRow, minCol, maxRow, maxCol = regions.bbox 
    #y0, x0, y1, x1 = regions.bbox
    
    regionHeight = maxRow - minRow
    regionWidth = maxCol - minCol
    rect_border = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor='red', linewidth=2, fill = False)
    ax1.add_patch(rect_border)

plt.show()

fig, ax1 = plt.subplots(1)
ax1.imshow(licensePlate, cmap="gray")
# the next two lines is based on the assumptions that the width of
# a license plate should be between 5% and 15% of the license plate,
# and height should be between 35% and 60%
# this will eliminate some
characterDimensions = (0.35*licensePlate.shape[0], 0.80*licensePlate.shape[0], 0.02*licensePlate.shape[1], 0.45*licensePlate.shape[1])
minHeight, maxHeight, minWidth, maxWidth = characterDimensions


characters = []
counter=0
columnList = []
for regions in regionprops(labelledPlate):
    minRow, minCol, maxRow, maxCol = regions.bbox 
    
    regionHeight = maxRow - minRow
    regionWidth = maxCol - minCol

    if regionHeight > minHeight and regionHeight < maxHeight and regionWidth > minWidth and regionWidth < maxWidth:
        roi = licensePlate[minRow:maxRow, minCol:maxCol]
        
        # draw a red bordered rectangle over the character.
        rect_border = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor='red', linewidth=2, fill = False)
        ax1.add_patch(rect_border)
    
        # resize the characters to 20X20 and then append each character into the characters list
        resizedChar = resize(roi, (20, 20))
        characters.append(resizedChar)

        # this is just to keep track of the arrangement of the characters
        columnList.append(minCol) 

plt.show()

