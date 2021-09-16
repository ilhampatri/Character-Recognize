import imutils
import numpy as np
import mahotas
import cv2 as cv
   
def load_fonts(datasetPath):
    data = np.genfromtxt(datasetPath, delimiter = ",",dtype = "uint8")
    target = data[:,2]
    data = data [:,12:].reshape(data.shape[0],20,20)
    return (data,target)
def deskew(image,width):
    
    (h, w) = image.shape[:2]
    moments = cv.moments(image)
        
    skew = moments["mu11"] / moments["mu02"]
    M =  np.float32 ([
         [1, skew, -0.1 * w * skew],
         [0,1,0]])
    image = cv.warpAffine(image, M, (w,h),
                            flags = cv.WARP_INVERSE_MAP | cv.INTER_LINEAR)
        
    image = imutils.resize(image,width)
        
    return image
    
def center_extent(image,size):
    (eW, eH) = size
        
    if image.shape[1] > image.shape[0]:
        image = imutils.resize(image,width =eW)
    else :
        image= imutils.resize(image, height =eH)
           
    extent =  np.zeros((eH,eW),dtype = "uint8")
       
    offsetX = (eW - image.shape[1]) // 2
    offsetY = (eH - image.shape[0]) // 2
    extent[offsetY:offsetY + image.shape[0], offsetX:offsetX + image.shape[1]] =image
        
    CM =  mahotas.center_of_mass(extent)
    (cY, cX) = np.round(CM).astype("int32")
    (dX, dY) =  ((size[0] // 2) - cX, (size[1] // 2)- cY)
    M = np.float32([[1,0,dX],[0,1,dY]])
    extent = cv.warpAffine(extent,M,size)
        
    return extent


