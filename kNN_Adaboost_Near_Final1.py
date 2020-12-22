#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 20:21:57 2020

@author: elvinagovendasamy
"""
#https://github.com/jamesajeeth/Data-Science/blob/master/Adaboost%20from%20scratch/Boosting%20Algorithm%20from%20scratch%20updated.ipynb

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
import math
from sklearn.model_selection import train_test_split


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
    #     classes={}
    #     for i in range(k):
    #         y_prediction=neighbours[i][-1]
    #         if y_prediction in classes:
    #             classes[y_prediction]+=1
    #         else:
    #             classes[y_prediction]=1
    #     # print(classes)
    #     sorted_classes=sorted(classes.items(),key=itemgetter(1),reverse=True)
    #     return sorted_classes[0][0] # On prend la plus grande proportion
    
    
    
    # # Je propose 2 autres methodes pour obtenir le même résultat
    # # METHODE 2:     
    #     # class_counter=Counter()
    #     # for n in neighbours:
    #     #     class_counter[n[-1]]+=1
    #     # return class_counter.most_common(1)[0][0]
    
    
    # # METHODE 3 :
        output=[]
        for i in range(len(neighbours)):
            output.append(neighbours[i][-1])
        prediction=max(set(output),key=output.count)
        return prediction
    
    
        
        
    # def manual_knn(dataset,num_splits,k):
    #     X_train,X_test,y_train,y_test=KNN.CV_KFold(num_splits,dataset)
    #     test_data,train_data=KNN.transforming_datasets(num_splits,dataset)
    #     predictions=[]
    #     for row in test_data:
    #         response=KNN.predict(train_data,row,k)
    #         predictions.append(response)
    #     return (predictions)
    
    
    def modified_manual_knn(X_train_data,y_train_data,k):
        predictions=[]
        
        # kf=KFold(n_splits=num_splits,random_state=1, shuffle=False)
        # X_b1=train_data.iloc[(train_data.shape[0]/num_splits):,:]
        # X_b2=train_data.iloc[(train_data.shape[0]/num_splits):(train_data.shape[0]/num_splits):,:]
        
        # for train_index, test_index in kf.split(train_data):
        #     X_train,X_test=X.iloc[train_index],X.iloc[test_index]
        #     y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
            
        # return X_train,X_test,y_train,y_test
        
        # return kf
        # print(kf)
        # for row in test_data:
        #     response=KNN.predict(train_data,row,k)
        #     predictions.append(response)
        # return (predictions)
        for i in range(len(X_train_data)):
            X_train_cv=X_train_data.iloc[-i:,:]
            y_train_cv=y_train_data.iloc[-i:,:]
            train_cv=np.concatenate([X_train_cv,y_train_cv],axis=1)
            X_test_cv=X_train_data.iloc[i,:]
            # y_test_cv=y_train_data.iloc[i,:]
            
            response=KNN.predict(train_cv,X_test_cv,k)
            predictions.append(response)
        return predictions
            
            
        
        
        
    
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
    
    
    def split_dataset(dataset,ratio):
            #X_train,X_test,y_train,y_test=KNN.CV_KFold(num_splits,dataset) 
            msk=np.random.rand(len(dataset)) < ratio
            train_data=dataset[msk]
            test_data=dataset[~msk]
            
            # X_train,X_test,y_train,y_test =train_test_split(X, y, test_size=0.33, random_state=42)
            
            # train_data=pd.concat([X_train,y_train],axis=1)
            # test_data=pd.concat([X_test,y_test],axis=1)
 
            # sample_train=[]
            # N_sample=round(ratio*train_data.shape[0]) # la taille de l'échantillon tiré du train
            
            # while len(sample_train) < N_sample: 
            #     index=randrange(len(train_data)) 
            #     sample_train.append(train_data.iloc[index,:])
        
            return train_data,test_data
            
        

            
       
    def sign(x):
        return abs(x)/x if x!=0 else 1
    
# On veut trouver le k optimal
   

if __name__=="__main__":

    path='winequality.csv'
    data=pd.read_csv(path,sep=';')
    
    Y=data.iloc[:,-1:]
    X=data.iloc[:,:-1]
    
    num_neighbours=np.arange(1,3)
    num_splits=3
    
    ratio=0.7
    # w_list=[]
    N=data.shape[0]
    # w=np.ones(N)/N 
    # w_list.append(w.copy()) 
    
    w=(np.ones(N)/N).tolist() 
    
    
    
    optimal_k=[]

    train_data,test_data=KNN.split_dataset(data,ratio)
    
    for i in range(1,3): # j'ai 2 bootstraps
        
        df_copy=train_data.sample(frac=1,replace=True).reset_index(drop=False)   # je prend tout mon échantillon train, mais c'est echantillon AVEC REMISE
        train_indice=df_copy['index']
        
        df_copy=df_copy.drop('index', 1)
        
        X_boot,y_boot=df_copy.iloc[:,:-1],df_copy.iloc[:,-1:]
        # X_test,y_test=test_data.iloc[:,:-1],test_data.iloc[:,-1:]
        
#         # Transformation en array pour faciliter le calcul de la distance euclideenne
        # test_data=test_data.astype(float).values.tolist()
        
        # X_boot1=X_boot.astype(float).values.tolist()
        # y_boot1=y_boot.astype(float).values.tolist()
        
#         # y_test is in dataframe format
#         # Creating appropriate formats for reading

#         # y_test_list=y_test.values.tolist() # sous format liste
        y_boot_list=y_boot['quality'].tolist() # sous format array
        # y_boot1=y_boot['quality'].to_numpy()
        
        
        n_train_boot=X_boot.shape[0] # la taille de mon echantillon train boostrap
#         n_test=y_test.shape[0] # la taille des données tests
        
#         # Maintenant on trouve le w_sample (le w de bootstrap)
        
        w_boot=[w[k] for k in train_indice]  
      
        
        
#         # w_sample_list.append(w.copy()) # on initialise la liste des poids
        
        errors=[]
        for k in range(1,2): 
            misclassified=[]
            y_pred=KNN.modified_manual_knn(X_boot,y_boot,k) # prediction de y_boot
            # print(y_pred)
            for i in range(len(y_pred)):
                misclassified.append(y_pred[i]!=y_boot_list[i])
                
            for i in range(len(misclassified)):
                if misclassified[i]==False:
                    misclassified[i]=1
                else:
                    misclassified[i]=0
            

            # erreur=np.average(misclassified, weights=w_boot, axis=0)
            erreur=0
            for i in range(len(w_boot)):
                erreur+=misclassified[i]*w_boot[i]
            
            errors.append([erreur,k])
            
            
 
        taux_erreur=sorted(errors, key=itemgetter(0)) # on réarrange les erreurs
            
# # #             # erreur minimum
        min_error=taux_erreur[0][0]
# #             # nombre de voisins associés à l'erreur minimum
        optimal_k.append(taux_erreur[0][1])
            
            
        errm = min_error/np.sum(w_boot)
        
        
        
        misclassified_optimal=[]
        k_optimal=taux_erreur[0][1]
        y_pred_optimal=KNN.modified_manual_knn(X_boot,y_boot,k_optimal)
        
        for i in range(len(y_pred_optimal)):
                misclassified_optimal.append(y_pred_optimal[i]!=y_boot_list[i])
                
        for i in range(len(misclassified_optimal)):
            if misclassified_optimal[i]==False:
                misclassified_optimal[i]=1
            else:
                misclassified_optimal[i]=0
        
        if errm==0:
            print('accuracy rate: ', 1)
        else:

            alpha = np.log((1-errm)/errm) 

            # on peut ajouter la condition que le sample_weight soit plus gros que 0 ou alpha est plus petit que 0: A VALIDER!!!!!!
            
            
            for i in range(len(misclassified_optimal)):
                w_boot[i]=w_boot[i]*np.exp(alpha*misclassified_optimal[i])

        
        
        train_indice_list=list(train_indice)

        for idx in range(len(train_indice_list)):
            w[train_indice_list[idx]]= w_boot[idx]
            

    y_predictions=[]    
    for items in optimal_k:
        y_predictions.append(KNN.modified_manual_knn(X_boot,y_boot,items))
    
    y_predictions=pd.DataFrame(y_predictions).T
    
   
    for i in range(len(y_predictions)):
        # for j in range(len(y_predictions.columns)):
        y_predictions['max']=max(set(y_predictions.iloc[i,:]),key=y_predictions.iloc[i,:].count)
       
    
        
        
    #     predictions_manual=0
    #     predictions_manual+=y_pred_list1[i,:]*alpha_list[i]
        
    #     # signA=np.vectorize(KNN.sign(predictions_manual))
    #     predictions_manual=np.sign(predictions_manual)

         
    # acc=np.sum(predictions_manual == y_test_array)/ len(y_test_array)
                
# #    
   
    

   
                
                
            
            
            
            
            
            
            



    