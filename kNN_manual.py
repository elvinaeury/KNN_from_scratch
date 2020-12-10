#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 20:21:57 2020

@author: elvinagovendasamy
"""

from sklearn import datasets
from scipy import spatial
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from operator import itemgetter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from collections import Counter
from scipy.spatial.distance import cdist
from random import seed
from random import random
from random import randrange
from sklearn import tree


class KNN:
    # calcul distance euclideenne de 2 vecteurs (x,y)
    def dist_euclid(x,y):
        distance=0.0
        for i in range(len(x)-1):
            distance+=((x[i]-y[i])**2)
            return np.sqrt(distance)
        
        
    def CV_KFold(nsplits,dataset):
        
        kf=KFold(n_splits=nsplits,random_state=1, shuffle=True)
    
        for train_index, test_index in kf.split(dataset):
            X_train,X_test=X.iloc[train_index],X.iloc[test_index]
            y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
            
        return X_train,X_test,y_train,y_test
    
    
    
    
    def transforming_datasets(nsplits,dataset):
        X_train,X_test,y_train,y_test=KNN.CV_KFold(nsplits,dataset)
        
        # Concatener les variables X et y dans chacun des 'training' et 'test' dataframes.
        train_data=pd.concat([X_train,y_train],axis=1)
        test_data=pd.concat([X_test,y_test],axis=1)
        
        # Transformation en array pour faciliter le calcul de la distance euclideenne
        test_data=test_data.astype(float).values.tolist()
        train_data=pd.DataFrame(train_data).astype(float).values.tolist()
        
        return test_data,train_data # sous forme de array
    
    
    
    def get_neighbours(train_dataset,test_row,k):
        distance=[]
        for idx in train_dataset:
            dist=KNN.dist_euclid(idx,test_row)
            distance.append((idx,dist))     
        distance=sorted(distance, key=itemgetter(1))[:k]
        
        # neighbours=[]
        # for i in range(k):
        #     neighbours.append(distance[i][0])
        neighbours=[n[0] for n in distance] # sortir les voisins correspondant aux k distances les plus petites
        return neighbours
    
    
    # Explication: Comme y est qualitatif, on affecte à chaque sous groupe 1 classe de y.
    # Nous cherchons à regarder l'erreur de classification usuelle (soit 1-accuracy score)
    # Regle de decision pour fabriquer le predicteur: 
        # on affecte la réponse 0 (idem 1, 2), s'il y a une plus grande proportion de 0(idem 1, 2)
    def predict(train_dataset,test_row,k):
        neighbours=KNN.get_neighbours(train_dataset,test_row,k)    
        classes={}
        for i in range(k):
            y_prediction=neighbours[i][-1]
            if y_prediction in classes:
                classes[y_prediction]+=1
            else:
                classes[y_prediction]=1
        
        sorted_classes=sorted(classes.items(),key=itemgetter(1),reverse=True)
        return sorted_classes[0][0] # On prend la plus grande proportion
    # Je propose 2 autres methodes pour obtenir le même résultat
    # METHODE 2:     
        # class_counter=Counter()
        # for n in neighbours:
        #     class_counter[n[-1]]+=1
        # return class_counter.most_common(1)[0][0]
    
    
    # METHODE 3 :
        # output=[]
        # for i in range(len(neighbours)):
        #     output.append(neighbours[i][-1])
        # prediction=max(set(output),key=output.count)
        # return prediction
    
    
        
        
    def manual_knn(dataset,num_splits,k):
        X_train,X_test,y_train,y_test=KNN.CV_KFold(num_splits,dataset)
        test_data,train_data=KNN.transforming_datasets(num_splits,dataset)
        
        predictions=[]
        for row in test_data:
            response=KNN.predict(train_data,row,k)
            predictions.append(response)
        return (predictions)
    
    def sklearn_kNN(dataset,num_splits,k):
        X_train,X_test,y_train,y_test=KNN.CV_KFold(num_splits,dataset)
        #test_data,train_data=transforming_datasets(num_splits,dataset)
        model=KNeighborsClassifier(k)
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        return y_pred
    
    
    def y_actual(dataset,num_splits,k):
        X_train,X_test,y_train,y_test=KNN.CV_KFold(num_splits,dataset)
        test_data,train_data=KNN.transforming_datasets(num_splits,dataset)
        actual=[]
        for row in test_data:
            y_true=row[-1]
            actual.append(y_true)
        return (actual)
    
    
    def accuracy_sklearn(dataset,num_splits,k):
        
        actual = KNN.y_actual(dataset,num_splits,k)
        predicted_sklearn=KNN.sklearn_kNN(dataset,num_splits,k)
        # predicted_manual_knn=manual_knn(dataset,num_splits,k)
        correct=0
        for i in range(len(actual)):
            if actual[i] == predicted_sklearn[i]:
                correct += 1
    	
        return correct / float(len(actual)) * 100.0
    
    
            
    
    def accuracy_manual_knn(dataset,num_splits,k):
        
        actual = KNN.y_actual(dataset,num_splits,k)
        # predicted_sklearn=sklearn_kNN(dataset,num_splits,k)
        predicted_manual_knn=KNN.manual_knn(dataset,num_splits,k)
        
        correct=0
        for i in range(len(actual)):
            if actual[i] == predicted_manual_knn[i]:
                correct += 1
    	
        return correct / float(len(actual)) * 100.0
    

        
       
    
# On veut trouver le k optimal
   

if __name__=="__main__":

    path='winequality.csv'
    data=pd.read_csv(path,sep=';')
    
    Y=data.iloc[:,-1:]
    X=data.iloc[:,:-1]
    
    num_neighbours=5
    num_splits=3
    

    X_train,X_test,y_train,y_test=KNN.CV_KFold(num_splits,data)
#    
    test_data,train_data=KNN.transforming_datasets(num_splits,data)
#    
    score_sklearn=KNN.accuracy_sklearn(data,num_splits,num_neighbours)
    score_manual_knn=KNN.accuracy_manual_knn(data,num_splits,num_neighbours)
    
    
    print('Accuracy score_manual : ',score_manual_knn)
    print('Accuracy score_sklearn: ',score_sklearn)
    

   
                
                
            
            
            
            
            
            
            



    