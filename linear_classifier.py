from PIL import Image
from numpy import *
from numpy import linalg as LA
import os
import matplotlib.pyplot as plot
import math
import sys

train_file = sys.argv[1]
test_file = sys.argv[2]

# Reading the training data and storing image path and label in imlist and classlist respectively
# Assigning index to each label/class
#------------------------------------------------------------------------------------------------
f = open(train_file,'rt')
lines = sort(f.readlines())
f.close()
imlist = [line.strip().split()[0] for line in lines]
classlist = [line.strip().split()[1] for line in lines]
imgDict = {}
j=0
for i in range(0, len(lines)):
    img,label = lines[i].strip().split()
    if label not in imgDict:
        imgDict[label] = j
        j+=1
num_class = len(imgDict)
num_train = len(imlist)

#------------------------------------------------------------------------------------------------

# Applying PCA on the training set and calculating the features array
#---------------------------------------------------------------------
immatrix = array([array(Image.open(img).convert('L').resize((64,64), Image.ANTIALIAS)).flatten()
              for img in imlist])

mean_X = mean(immatrix, axis=0)
immatrix = immatrix - mean_X
U,S,V = linalg.svd(immatrix)
V = V.T
Coeff = dot(immatrix, V)
Coeff = Coeff[:, :33]
Coeff[:, 32] = 1

#----------------------------------------------------------------------

# Calculating weight matrix using linear classifier equation
#------------------------------------------------------------
w = random.random((num_class, 33))
eta = 0.00001
for i in range(0,3000):
    wtemp = zeros((num_class, 33))
    for j in range(num_train):
        label = classlist[j]
        prod = array(dot(w, Coeff[j]), dtype=float64)
        large = max(prod)
        prod = prod-large
        den = sum(math.e**prod)
        num = math.e**(dot(w[imgDict[label], :], Coeff[j])-large)
        prob = num/den
        wtemp[imgDict[label], :] += (1-prob)*Coeff[j]
    w = add(w, eta*wtemp)

#-------------------------------------------------------------

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
Scoeff = Scoeff[:, :33]
Scoeff[:, 32] = 1

#----------------------------------------------------------------------

# Multiplying weight matrix with each sample coeff matrix 
# and getting class with highest probability
#------------------------------------------------------------
for sample in Scoeff:
    probMatrix = dot(w, sample)
    tp = argmax(probMatrix)
    for label, ind in imgDict.items():
        if  ind == tp:
            print(label)

#-------------------------------------------------------------

