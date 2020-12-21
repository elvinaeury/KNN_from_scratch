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
dataset=pd.read_csv(path,sep=';')



def dist_euclid(x,y):
    distance=0.0
    for i in range(len(x)-1):
        distance+=((x[i]-y[i])**2)
        return np.sqrt(distance)


# def subsample(data,ratio):
#     sample=[]
#     N=round(len(dataset)*ratio) # On peut faire notre sous-echantillon à partir d'une partie des données
#     while len(sample) < N: # tant que la taille de l'échantillon est plus petit que N
#         index=randrange(len(dataset)) # prend une valeur au hasard et avec remise
#         sample.append(dataset.iloc[index,:])
#     return pd.DataFrame(sample)

# data_sample=subsample(dataset,0.2)

ratio=0.3
folds=3
iterations=10

# sample_=subsample(dataset,ratio)

def erreur_boostrap(data,k,ratio,folds,iterations):
    
    predictions=[]
    fold = []
    dataset_split = []
    distance=[]
    actual=[]
    errors=[]
    

    for i in range(iterations): # le nombre d'échantillons = B
    
        df_copy = data.sample(frac=0.3).reset_index(drop=True) 
        fold_size = int(df_copy.shape[0] / folds)
        
        # for loop to save each fold
        for i in range(folds):
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
        for i in range(folds):
            r = list(range(folds))
            r.pop(i)
            for j in r :
                if j == r[0]:
                    train_data = dataset_split[j] 
                else:    
                    train_data=np.concatenate((train_data,dataset_split[j]), axis=0)
        
        # On a le test/ validation set (1 fold)    
            test_data=np.array(dataset_split[i])# le nombre de colonne
               
            # train_data=pd.DataFrame(train_data)
            print(train_data)
        #neighbours = []    
        for row in test_data:
            # print(row)
            for idx in train_data:
                # print(idx)
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
    return erreur_moyenne



ratio=0.3
folds=3
iterations=10

#mean_error_boostrap=erreur_boostrap(k, ratio, folds, iterations)


erreurs_k=[]
for k in range(1,10):
    erreurs_k.append((erreur_boostrap(dataset,k, ratio, folds, iterations),k))
    
min_error=sorted(erreurs_k, key=itemgetter(0))


# AMELIORATION:
     # PLOT LE GRAPH LE MIN_ERROR ET K
print("l'erreur moyenne est : ", min_error[0])     
# print("le accuracy score est : ", 100-erreur_moyenne)

