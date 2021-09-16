##from sklearn.externals import joblib\n",
##from sklearn.svm import LinearSVC\n",
##from pyimagesearch.hog import HOG\n",
##from pyimagesearch import dataset\n",
##import argparse\n",
#%run train.py --dataset datasets/train.csv --model models/svm.cpickle

import joblib
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from hog import HOG
import dataset

(fonts, target) = dataset.load_fonts('datasets/Huruf Revisi.csv')



data=[]
print(type(data))
hog = HOG(orientations = 9, pixelsPerCell = (8,8),cellsPerBlock = (2,2),
          transform= True,blocknorm="L2-Hys")

for image in fonts :
    image = dataset.deskew(image,20)
    image = dataset.center_extent(image, (20,20))

    hist = hog.describe(image)
    data.append(hist)

print(hist)
rfc = RandomForestClassifier(
    n_estimators=150)
rfc.fit(data,target)
#model = LinearSVC(random_state =42)
#model.fit(data,target)
#joblib.dump(model, 'models/rfc.cpickle')
joblib.dump(rfc, 'models/rfc.cpickle')
