# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 18:34:14 2016

@author: Marcia
"""



"""
Naive Bayes Classifier on NC School Dataset
"""
import csv
import numpy
import pandas as pd
#Read in csv data as a list of strings
filename ='/Users/Marcia/OneDrive/DSBA 6156 ML/CMS Project/DataFiles For Python/Final_161120.csv'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data, delimiter=',',quoting=csv.QUOTE_NONE)
x = list(reader)
#Turn list into a numpy array (still all strings)
all_data_strings = numpy.array(x)
#Generate and print list of feature names, ie column headers, ie the first row.
feature_names=all_data_strings[0:1,:]
print(feature_names)
#Generate an array of all rows after the header row, keep as strings.
dataset_strings = all_data_strings[1:,:] 
#Convert features from strings to numbers
dataset = dataset_strings.astype('float')
#print(dataset.shape)
#Print list of only the features used in this model run.
print()
print()
print("Features we are keeping in this model version:")
#Choose the columns you want to use as feature inputs.
print(feature_names[0:1,38:50])#PCA
#print(feature_names[0:1,11:12])#EDSPer
#print(feature_names[0:1,[7,8,10,11,16,19,21,25,26,29,31]])#SMExpert
#print(feature_names[0:1,[7,8,10,16,19,21,25,26,29,31]])#SMExpert_NoEDSPer
for i in range(1, 2):
    print()
    print("NB Classifier With Continous Feature Set.")
    print("Sample Number: %d" %i )
 #Shuffle ie randomly reorder the dataset
    numpy.random.shuffle(dataset)
#Define sample as first 1000 rows of the array called 'dataset'
    sample=dataset[0:500,:]
     #print(sample.shape)
    X = sample[:,38:50]#PCA
 #   X = sample[:,11:12]#EDSPer
#    X = sample[:,[7,8,10,11,16,19,21,25,26,29,31]]#SMExpert
 #   X = sample[:,[7,8,10,16,19,21,25,26,29,31]]#SMExpert_NoEDSPer
    Y = sample[:,0]

    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    y_pred = gnb.fit(X,Y).predict(X)
#print("Number of mislabeled points out of a total %d points : %d. The accuracy for labled points is %d percent"
# % (X.shape[0],(Y != y_pred).sum(), (int(X.shape[0]-(Y!= y_pred).sum())*100/(X.shape[0]))))
    print("Number of mislabeled points out of a total %d points: %d."
    % (X.shape[0],(Y != y_pred).sum()))
    print("The accuracy for labled points is %d percent." % (int(X.shape[0]-(Y!=
    y_pred).sum())*100/(X.shape[0])))
    print()
    from sklearn.metrics import classification_report 
    print(classification_report(Y,y_pred))
    from sklearn.metrics import confusion_matrix
    confusion_matrix=confusion_matrix(Y, y_pred)
    #print()
    print("Confusion Matrix")
    print("  TP  FP")
    print("  FN  TN")
    print(confusion_matrix)
    
    df=pd.DataFrame({"Y_Actual" : Y, "Y_Pred" : y_pred})
    df.to_csv('/Users/Marcia/OneDrive/DSBA 6156 ML/CMS Project/DataFiles For Python/NB_Pairs_PCA.csv', index=False)