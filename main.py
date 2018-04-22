#!/usr/bin/python
# -*- coding:utf-8 -*-
# Apache License Version 2
# author: zhaofeng-shu33
# file-description: detect outliers of students body test data
# Last modified on 23/04/2018 
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt 
from gi import GraphBasedInterpolation
class test_bench:
    '''
    class used to compare different classification method
    '''
    def __init__(self):
        self.data = pd.read_excel('data_preprocessing.xlsx') # the data file is manually processed and renamed before given to program.
        self.data = self.data[0:self.data.shape[0]-1]# remove the last record, which has five 'NA'
        data_tmp = self.data.loc[:,self.data.columns[2]:self.data.columns[-1]] # remove gender and age, which are category values
        scaler = MinMaxScaler()# default feature_range =[0,1]
        self.scaled_data = scaler.fit_transform(data_tmp) 
        self.gender = self.data.loc[:,self.data.columns[0]].values>0.5 # 1 is girl
        return
    def classifier(self,method_list=['lda']):
        # gender is label, use scaled_data to predict gender  
        kf = KFold(n_splits = 5,shuffle = True)
        turn_index = 1
        error_cnt = 0
        for method in method_list:
            for train_index,test_index in kf.split(self.gender):
                predicted_gender=None
                if(method=='lda'):
                    lda = LinearDiscriminantAnalysis()
                    lda.fit(self.scaled_data[train_index],self.gender[train_index])
                    predicted_gender = lda.predict(self.scaled_data[test_index])
                elif(method=='gi'):
                    gi = GraphBasedInterpolation(modify_weight=True)
                    gi.fit(self.scaled_data)
                    predicted_gender = (gi.predict(train_index,test_index,self.gender[train_index])>0.5)
                else:
                    raise(Exception("method %s not implemented" % method))
                print("Use method %s..." % method)
                # collect two types of errors
                # since the number of boys is larger than the number of girls
                # P_e(boy | girl) and P_e(girl | boy)
                comparision_result = predicted_gender == self.gender[test_index]
                error_case = np.where(np.logical_not(comparision_result))
                error_case_index = test_index[error_case]
                for i in error_case_index:
                    print('Turn ',turn_index,self.gender[i],'gender_index',i)
                turn_index += 1
                error_cnt +=len(error_case_index)
                # as can be seen, use lda, the error is very small, it means that we can use the body data to discriminate gender very well
            print('Average error rate: ',error_cnt*1.0/self.data.shape[0])
        return

if __name__  == '__main__':
    new_test_bench = test_bench()
    new_test_bench.classifier(method_list=['lda','gi'])


