#!/usr/bin/python
# -*- coding: utf-8 -*-
# author: zhaofeng-shu33
# license: Apache License Version 2
# implementation of graph based interpolation(no distinguish between train and test)
# exported class: GraphBasedInterpolation
# exported method: fit (generate similar matrix from feature matrix), 
#                  predict(give a graph signal function which only has value on partial nodes of graph, predict its value on the remaining nodes)
# Created on 2018-04-21

import numpy as np
import scipy.sparse.linalg

class GraphBasedInterpolation(object):
    def __init__(self,modify_weight=False):
        self.weight_matrix = None
        self.laplace_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.degree = None
        self.modify=modify_weight # whether use RBF kernel to scale the weight matrix
    def fit(self,data):#data is normalized numpy array
        # calc_laplace_matrix from normalized data
        # num of rows = instance num, every node in the graph is an instance
        # num of columns = feature num
        # d=1/(norm_2(a-b)^2+epsilon)
        instance_num = data.shape[0]
        # data_square = np.sum(data*data,axis=1)
        # self.weight_matrix = np.dot(data,data.T)/np.kron(data_square,data_square).reshape(instance_num,instance_num), cosine similarity not works
        data_square = np.reshape(np.sum(data*data,axis=1),[instance_num,1]) # sum over columns
        self.weight_matrix = 1/(1e-3+(data_square + data_square.T) -2*np.dot(data,data.T)) # np broadcast, euclid distance
        np.fill_diagonal(self.weight_matrix,np.zeros(instance_num))
        self._compute_laplace()
    def _compute_laplace(self):
        '''
        compute normalized laplace matrix from weight matrix

        return: none
        '''
        self.degree = np.sum(self.weight_matrix,axis=1)
        self.laplace_matrix = np.diag(self.degree)-self.weight_matrix
        self.laplace_matrix = np.dot(np.dot(np.diag(1/np.sqrt(self.degree)),self.laplace_matrix),np.diag(1/np.sqrt(self.degree))) #normalization
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.laplace_matrix)

    def predict(self, train_index, test_index, partial_signal_function):#graph interpolation codes goes here
        '''
        predict signal_function on test_index

        train_index : known instance index
        test_index: unknown instance index
        partial_signal_function: signal value at known instance index(train_index)        
        return: predicted signal value at unknown instance index(test_index)
        '''
        assert(len(train_index)==len(partial_signal_function))
        if(self.modify):
            # first extract the submatrix
            weight_matrix_tmp = self.weight_matrix[np.ix_(train_index,train_index)]
            # RMF scaling
            theta = np.mean(partial_signal_function)
            psf_m = partial_signal_function.reshape([len(partial_signal_function),1]).astype(float) # partial_signal_function_matrix
            self.weight_matrix[np.ix_(train_index,train_index)] = weight_matrix_tmp*np.exp(-np.power(psf_m-psf_m.T,2)/(theta**2)) # np broadcast
            self._compute_laplace()
        # construct S^C matrix
        S=train_index
        SC=test_index
        SC_Matrix=np.dot(self.laplace_matrix,self.laplace_matrix).take(np.asarray(SC,int),axis=0).take(np.asarray(SC,int),axis=1)

        ws=abs(scipy.sparse.linalg.eigs(SC_Matrix,k=1,which='SM',return_eigenvectors=False)) #return the smallest eigenvalue
        UK=self.eigenvectors.T[self.eigenvalues<ws]
        UKS=UK.take(np.asarray(S,int),axis=1).T
        UKSC=UK.take(np.asarray(SC,int),axis=1).T
        #assemble_laplace_matrix f on S        
        fS=partial_signal_function
        gS=np.sqrt(self.degree[np.asarray(S,int)])*fS        
        #least square interpolation to find gSC
        gSC=np.dot(UKSC,np.linalg.lstsq(UKS,gS,rcond=None)[0])
        # assemble_laplace_matrix original S
        g=np.zeros(self.laplace_matrix.shape[0])
        g.put(S,gS)
        g.put(SC,gSC)
        hat_f =(1/np.sqrt(self.degree))*g
        return hat_f[test_index]


