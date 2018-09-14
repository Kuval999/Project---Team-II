# -*- coding: utf-8 -*-
"""
Find license plate in given car image
Steps - 
1. Read image
2. Convert it into gray scale(pixels - 0 to 255)
3. Convert it into binary image (pixels - black(255) or white(0))
4. Use connected component analysis(CCA) - identifies and groups connected regions on the foreground
   Note - Connected region -> adjacent pixels have same value 
5. Draw rectangular boxes over connected regions
6. Apply filter to detect number plate
    a. Assumption - Ratio of width and length of license plate to full image
                  - Number plate area 
    NOT DONE -  b. Vertical projection - adds all pixel values in each column. sum of these pixel values will be more due to fact that characters are written on it
"""

from skimage import io, filters, measure, morphology
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    #image_path = input("Enter car image path : ")
    imagePath = 'workingImgs\k1.jpg'
    carImage = io.imread(imagePath)
    grayCarImage = io.imread(imagePath)
    print(grayCarImage)    
    #read image in gray scale
    grayCarImage = io.imread(imagePath, as_grey = True)
    print(grayCarImage)
    #grey scale image has to be in 2-dimension 
    print('Image shape : ', grayCarImage.shape)
    #print(grayCarImage)
    
    #image pixels value range from 0 to 1. Multiply each pixel values with 255, if you want pixel range from 0-255.
    grayCarImage = grayCarImage * 255
    print(grayCarImage.max()) 
    print(grayCarImage.min())
    print(grayCarImage)
    
    #convert image into binary image - use threshold-otsu
    thresholdValue = filters.threshold_otsu(grayCarImage)
    binaryCarImage = grayCarImage > thresholdValue
    print(binaryCarImage)
    binaryCarImage = morphology.opening(binaryCarImage, morphology.rectangle(4,4))
    
    #plot images
    plt.subplot(221)
    plt.title("Original image")
    plt.imshow(carImage)
    plt.subplot(222)
    plt.title("Gray scale image")
    plt.imshow(grayCarImage, 'gray')
    plt.subplot(223)
    plt.title('Binary image')
    plt.imshow(binaryCarImage, 'gray')
    plt.show()
    
    fig, (ax1) = plt.subplots(1)
    ax1.imshow(grayCarImage, 'gray')
    
    #find connected regions on the image
    connectedRegion = measure.label(binaryCarImage)
    #print(connectedRegion)
    
    #regionprops - creates properties of all connected regions like area, bounding box, label etc
    for region in measure.regionprops(connectedRegion):
        
        if (region.area < 80000 ):
            #assumption - area of number plate is more than 1000 and less than 8000
            continue
        #bbox return box coordinates of connected region
        minRow, minCol, maxRow, maxCol = region.bbox

        #draw rectangular box on each connected region 
        rectBorder = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor='red', linewidth=2, fill = False)
        ax1.add_patch(rectBorder)
    
    plt.show()
    
    #assumptions - number plate 
    #width > height
    #proportion - width of number plate to full image - 20% to 40%
    #proportion - height of number plate to full image - 8% to 20%
    print(connectedRegion.shape[0])
    print(connectedRegion.shape[1])
    
    #set max width, height and min width, height a license plate can have 
    minHeight = 0.05*connectedRegion.shape[0]    
    maxHeight = 0.10*connectedRegion.shape[0]
    minWidth = 0.20*connectedRegion.shape[1]
    maxwidth = 0.7*connectedRegion.shape[1]

    
    fig, (ax1) = plt.subplots(1)
    ax1.imshow(grayCarImage, 'gray')
    
    
    plateLikeObjects = []
    plateLikeObjCoordinate = []    
    
    count = 0
    for region in measure.regionprops(connectedRegion):
        if(region.area < 80000):
            continue
        
        minRow, minCol, maxRow, maxCol = region.bbox
        regionHeight = maxRow - minRow
        regionWidth = maxCol - minCol
        
        if regionHeight >= minHeight and regionHeight <= maxHeight and regionWidth >= minWidth and regionWidth <= maxwidth and regionWidth > regionHeight and minRow > 1600:
             #keep track of how many objects are detected as number plates
             count += 1 
             #print(minRow, minCol, maxRow, maxCol)
             #store objects detected as number plate in an array
             
             plateLikeObjects.append(binaryCarImage[minRow:maxRow, minCol:maxCol])
             plateLikeObjCoordinate.append((minRow, minCol, maxRow, maxCol))
             
             rectBorder = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor='red', linewidth=2, fill = False)
             ax1.add_patch(rectBorder)
    
    plt.show()
    
    
except FileNotFoundError:
    print("Sorry! No such image found!")























