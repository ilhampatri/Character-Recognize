##from __future__ import print_function\n",
##from sklearn.externals import joblib\n",
##from pyimagesearch.hog import HOG\n",
##from pyimagesearch import dataset\n",
##import argparse\n",
##import mahotas\n",
##import cv2\n",
#%run classify.py --model models/svm.cpickle --image images/tes.jfif

from __future__ import print_function
import joblib
from sklearn.svm import LinearSVC
from hog import HOG
import dataset
import mahotas
import cv2 as cv
import imutils
from gtts import gTTS
import os
import vlc
from time import sleep
from picamera import PiCamera
#camera = PiCamera()
#camera.start_preview()
#camera.rotation=-90
#camera.resolution = (512,64)
#camera.resolution = (768,64)

#camera.resolution = (1024,64)
#sleep(7)
#camera.capture('/home/pi/HURUF/images/image.png')                   
#camera.capture('/home/pi/assistant-sdk-python/google-assistant-sdk/googlesamples/assistant/grpc/HURUF/images/image.png')
                    #camera.capture('/home/pi/assistant-sdk-python/google-assistant-sdk/googlesamples/assistant/grpc/HURUF/images/Arial/3.png')
                    
#camera.stop_preview()
language = "id"
slow_audio_speed =False
filename = "angka.mp3"
model = joblib.load('models/rfc.cpickle')

hog = HOG(orientations = 9 , pixelsPerCell=(8,8),
    cellsPerBlock = (2,2), transform = True,blocknorm="L2-Hys")

image =  cv.imread("images/image.png")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

blurred = cv.GaussianBlur(gray,(5,5), 0)
edged = cv.Canny(blurred,30,90)
cnts, _ = cv.findContours(edged.copy(),cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE)

cnts = sorted([(c,cv.boundingRect(c)[0]) for c in cnts] ,
             key = lambda x:x[1])

huruf = []

for (c, _) in cnts :
       
        
        (x,y,w,h) = cv.boundingRect (c)    
        if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
        #if (w>=4 and h>=8):
              roi = gray[y:y+h,x:x+w]
              thresh = roi.copy()
              T = mahotas.thresholding.otsu(roi)
              thresh[thresh>T] = 255
              thresh = cv.bitwise_not(thresh)
        
              thresh = dataset.deskew(thresh,20)
              thresh = dataset.center_extent(thresh, (20,20))
        
              cv.imshow("thresh",thresh)
              hist = hog.describe(thresh)
              font = model.predict([hist])[0]
              if font==97:
                  font="a"
              if font==98:
                  font="b"
              if font==99:
                  font="c"
              if font==100:
                  font="d"
              if font==101:
                  font="e"
              if font==102:
                  font="f"
              if font==103:
                  font="g"
              if font==104:
                  font="h"
              if font==105:
                  font="i"
              if font==106:
                  font="j"
              if font==107:
                  font="k"
              if font==108:
                  font="l"
              if font==109:
                  font="m"
              if font==110:
                  font="n"
              if font==111:
                  font="o"
              if font==112:
                  font="p"
              if font==113:
                  font="q"
              if font==114:
                  font="r"
              if font==115:
                  font="s"
              if font==116:
                  font="t"
              if font==117:
                  font="u"
              if font==118:
                  font="v"
              if font==119:
                  font="w"
              if font==120:
                  font="x"
              if font==121:
                  font="y"
              if font==122:
                  font="z"
              if font==65:
                  font="A"
              if font==66:
                  font="B"
              if font==67:
                  font="C"
              if font==68:
                  font="D"
              if font==69:
                  font="E"
              if font==70:
                  font="F"
              if font==71:
                  font="G"
              if font==72:
                  font="H"
              if font==73:
                  font="I"
              if font==74:
                  font="J"
              if font==75:
                  font="K"
              if font==76:
                  font="L"
              if font==77:
                  font="M"
              if font==78:
                  font="N"
              if font==79:
                  font="O"
              if font==80:
                  font="P"
              if font==81:
                  font="Q"
              if font==82:
                  font="R"
              if font==83:
                  font="S"
              if font==84:
                  font="T"
              if font==85:
                  font="U"
              if font==86:
                  font="V"
              if font==87:
                  font="W"
              if font==88:
                  font="X"
              if font==89:
                  font="Y"
              if font==90:
                  font="Z"
              if font==166:
                  font=" "
              else:
                  font=="NOT DETECTED"
              print("Huruf : {}".format(font))     
              huruf_baru = ("{}".format(font))
              huruf.append(huruf_baru)
              cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
              cv.putText(image, str(font), (x - 10 , y - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 1)
              cv.imshow("image",image)
              cv.waitKey(0)
def convert(huruf):
    str1 = ""
    return(str1.join(huruf))

print ("Huruf lengkapnya adalah : {} ".format(huruf))
baru=(convert(huruf))
print(convert(huruf))

#audio_created = gTTS(text=baru, lang=language, slow=slow_audio_speed)
#audio_created.save(filename)
#media = vlc.MediaPlayer(filename)
#media.play()

cv.imwrite('image.jpg',image)
  


