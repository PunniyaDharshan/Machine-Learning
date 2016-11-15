# -*- coding: utf-8 -*-
"""
Naive Bayes Classifier on NC School Dataset
"""
import csv
import numpy

#our big problem was using 'rb' below, i changed to 'r'; the example was wrong!
raw_data = open('E:/UNCC/Fall 16/Machine Learning/project/Sample_Data_161101_JustLunch_200.csv', 'rU')
#I don't remember where I found the following code, but it works.
reader = csv.reader(raw_data)
data=[]
for line in reader:
    data.append(line)
dataset=numpy.array(data).astype('float')

#x = list(dataset)

#print(dataset.shape)

numpy.random.shuffle(dataset)
training = dataset[:120,:]
test=dataset[120:,:]
# separate the data from the target attributes
# using X=0:14 somehow made us miss the last column, its a python thing.
# but X and Y are correct as defined below, as I double checked against the csv
X = training[:,0:1]
X=numpy.array(X,dtype=float)
Y = training[:,1]
Y=numpy.array(Y,dtype=int)
Xtest=test[:,0:1]
Xtest=numpy.array(Xtest,dtype=float)
print(Xtest)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
#I had ".predict(Y) for the longest time, produced error, that was my mistake when I put in X,Y for the Iris.XXX stuff
model=gnb.fit(X,Y).predict(Xtest)
print((Xtest.shape[0],model,Y))
dup=model.copy()
max_index=dup.size
v=0
d=0
li=list()
while v<max_index:
    li.append(v)
    v=v+1
for i in li:
    if(model[i]==Y[i]):
         d=d+1
print((d))        
         
   
    
    
    
        
