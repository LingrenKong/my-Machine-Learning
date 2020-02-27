# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:54:22 2020

@author: 1551086871@qq.com, Lingren Kong, github: https://github.com/LingrenKong
"""
import numpy as np

class KMeans():
    """KMeans算法
    
    Parameters
    ----------
    n_clusters:int, default=8
    The number of clusters to form as well as the number of centroids to generate.
    聚类目标数量

    n_init:int, default=10
    Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
    不同初始化的次数

    max_iter:int, default=300
    Maximum number of iterations of the k-means algorithm for a single run.
    算法最大迭代次数

    tol:float, default=1e-4
    Relative tolerance with regards to inertia to declare convergence.
    收敛判定的界限

    """
    def __init__(self,n_clusters=8,n_init=10, max_iter=300, tol=0.0001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init

    def _check_fit_data(self, X):
        """监测样本量是不是比K大"""
        try:
            size_of_data = X.shape[0]
        except:
            raise ValueError(f"X是{type(X)}而不是一个numpy的ndarray")
        if size_of_data<self.n_clusters:
            raise ValueError(f"样本量{size_of_data}比K（{self.n_clusters}）小了……")
        return size_of_data
    def fit(self,X):
        """拟合"""
        size_of_data = self._check_fit_data(X)#先检查一下
        for _ in range(self.n_init):
            init_choice = np.random.choice(size_of_data, self.n_clusters,replace=False)#不放回抽样
            init_points = X[init_choice]

