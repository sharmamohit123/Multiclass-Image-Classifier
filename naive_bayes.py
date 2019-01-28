from PIL import Image
from numpy import *
from numpy import linalg as LA
import os
import matplotlib.pyplot as plot
import math
import sys

train_file = sys.argv[1]
test_file = sys.argv[2]

# Reading the training image paths ans storing in imlist
# Also keeping a count of the training data per class using imgDict 
#------------------------------------------------------------------
f = open(train_file,'rt')
lines = sort(f.readlines())
f.close()
imlist = [line.strip().split()[0] for line in lines]
imgDict = {}
for i in range(0, len(lines)):
    img,label = lines[i].strip().split()
    imgDict[label] = i
    
#--------------------------------------------------------------------

# Applying PCA on the training set and calculating the features array
#---------------------------------------------------------------------
immatrix = array([array(Image.open(img).convert('L').resize((64,64), Image.ANTIALIAS)).flatten()
              for img in imlist])

mean_X = mean(immatrix, axis=0)
immatrix = immatrix - mean_X
U,S,V = linalg.svd(immatrix)
V = V.T
Coeff = matmul(immatrix, V)
Coeff = Coeff[:, :32]

#-----------------------------------------------------------------------

# Calculating mean and variance matrix for each class
#-------------------------------------------------------
meanDict = {}
varDict = {}
s=0
for label in sorted(imgDict, key=imgDict.get):
    e = imgDict[label]+1
    meanDict[label] = array(mean(Coeff[s:e, :], axis=0))
    varDict[label] = array(var(Coeff[s:e, :], axis=0))
    s = e

#--------------------------------------------------------

# Getting the Coeff matrix for test data using training mean and coeff
#---------------------------------------------------------------------
f = open(test_file,'rt')
lines = f.readlines()
f.close()
samplelist = [line.strip() for line in lines]

smatrix = array([array(Image.open(img).convert('L').resize((64,64), Image.ANTIALIAS)).flatten()
              for img in samplelist])

smatrix = smatrix - mean_X
Scoeff = dot(smatrix, V)
Scoeff = Scoeff[:, :32]

#----------------------------------------------------------------------

# Calculating normal probability distribution for each test image for every class 
# Finding the class having maximum probability
#-----------------------------------------------------------------------------
probDict = {}
for sample in Scoeff:
    for label in sorted(imgDict, key=imgDict.get):
        A = (2*math.pi*varDict[label])**0.5
        exp = array(((sample-meanDict[label])**2)/(2*varDict[label]), dtype=float64)
        prob = array((math.e**(-1.0*exp))/A, dtype=float64)
        probDict[label] = prod(prob)
    print(sorted(probDict, key=probDict.get, reverse=True)[0])

#---------------------------------------------------------------------------------