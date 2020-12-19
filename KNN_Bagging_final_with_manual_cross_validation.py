#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 20:48:16 2020

@author: elvinagovendasamy
"""

import pandas as pd
import numpy as np
from random import randrange
from operator import itemgetter

path='winequality.csv'
data=pd.read_csv(path,sep=';')

folds=4
# for i in range(folds):
#     sample_index=np.random.choice(data.index,len(data)) # resample with replacement
#     data_sample=data.loc[sample_index]
#     print(data_sample)
#     y_bs=data_sample.iloc[:,-1:]
#     X_bs=data_sample.iloc[:,:-1]        
def cross_validation_split(dataset, folds):
        dataset_split = []
        df_copy = dataset
        fold_size = int(df_copy.shape[0] / folds)
        
        # for loop to save each fold
        for i in range(folds):
            fold = []
            # while loop to add elements to the folds
            while len(fold) < fold_size:
                # select a random element
                r = randrange(df_copy.shape[0])
                # determine the index of this element 
                index = df_copy.index[r]
                # save the randomly selected line 
                fold.append(df_copy.loc[index].values.tolist())
                # delete the randomly selected line from
                # dataframe not to select again
                df_copy = df_copy.drop(index)
            # save the fold     
            dataset_split.append(np.asarray(fold))
            
        return dataset_split 

def dist_euclid(x,y):
    distance=0.0
    for i in range(len(x)-1):
        distance+=((x[i]-y[i])**2)
        return np.sqrt(distance)




k=4
ratio=0.3
f=4
sample=[]
predictions=[]
fold = []
dataset_split = []
distance=[]
actual=[]
errors=[]
size=20
for i in range(size): # la taille de mon echantillon boostrap = 100
    N=round(len(data)/size) # On peut faire notre sous-echantillon à partir d'une partie des données
    while len(sample) < N: # tant que la taille de l'échantillon est plus petit que N
        index=randrange(len(data)) # prend une valeur au hasard et avec remise
        sample.append(data.iloc[index,:])
    sample=pd.DataFrame(sample)

    df_copy = data
    fold_size = int(df_copy.shape[0] / f)
    
    # for loop to save each fold
    for i in range(f):
        # while loop to add elements to the folds
        while len(fold) < fold_size:
            # select a random element
            r = randrange(df_copy.shape[0])
            # determine the index of this element 
            index = df_copy.index[r]
            # save the randomly selected line 
            fold.append(df_copy.loc[index].values.tolist())
            # delete the randomly selected line from
            # dataframe not to select again
            df_copy = df_copy.drop(index)
        # save the fold     
        dataset_split.append(np.asarray(fold))

   

# On trouve le training set (avec k-1 folds)
    for i in range(f):
        r = list(range(f))
        r.pop(i)
        for j in r :
            if j == r[0]:
                train_data = dataset_split[j] 
            else:    
                train_data=np.concatenate((train_data,dataset_split[j]), axis=0)
    
    # On a le test/ validation set (1 fold)    
        test_data=np.array(dataset_split[i][:,0:12])# le nombre de colonne
              
        # train_data=pd.DataFrame(train_data).astype(float).values.tolist()    
        
    for row in test_data:
        for idx in train_data:
            dist=dist_euclid(idx,row)
            distance.append((idx,dist))     
        distance=sorted(distance, key=itemgetter(1))[:k]
        
        neighbours=[n_[0] for n_ in distance] 

        classes={}
        for i in range(k):
            y_prediction=neighbours[i][-1]
            if y_prediction in classes:
                classes[y_prediction]+=1
            else:
                classes[y_prediction]=1
        
        sorted_classes=sorted(classes.items(),key=itemgetter(1),reverse=True)
        response=sorted_classes[0][0]
        predictions.append(response)
        
    for row in test_data:
        y_true=row[-1]
        actual.append(y_true)
        
    incorrect=0
    for i in range(len(actual)):
        if actual[i] != predictions[i]:
            incorrect += 1
    
    error=(incorrect / float(len(actual)) * 100.0)
           
    errors.append(error)
               
erreur_moyenne=np.mean(errors) # of cross validation and bagging
print("l'erreur moyenne est : ", erreur_moyenne)     
print("le accuracy score est : ", 100-erreur_moyenne) 