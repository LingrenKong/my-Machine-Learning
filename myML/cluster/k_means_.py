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
    def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=0.0001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.y_label = None

    def _check_fit_data(self, X):
        """监测样本量是不是比K大"""
        try:
            size_of_data = X.shape[0]
        except:
            raise ValueError(f"X是{type(X)}而不是一个numpy的ndarray")
        if size_of_data<self.n_clusters:
            raise ValueError(f"样本量{size_of_data}比K（{self.n_clusters}）小了……")
        return size_of_data

    def fit(self, X):
        """拟合"""
        size_of_data = self._check_fit_data(X)#先检查一下
        modelset = []
        for _ in range(self.n_init):
            score = False
            init_choice = np.random.choice(size_of_data, self.n_clusters, replace=False)#不放回抽样
            centroid_points = X[init_choice,:]
            print("初始化中心点为：",centroid_points)
            y_label = np.zeros((X.shape[0], 1)) #分类标签初始化
            dist = np.zeros((X.shape[0], self.n_clusters)) #距离矩阵初始化
            for _ in range(self.max_iter):
                for i in range(self.n_clusters):
                    temp = X-centroid_points[i, :]
                    dist[:,i] =np.sum(temp**2,axis=1)
                y_label = np.argmin(dist, axis=1)
                print("分配的类标标签为\n", y_label)
                for i in range(self.n_clusters):
                    centroid_points[i, :] = np.mean(X[y_label==i],axis=0)
                print("重新根据均值分配中心点\n",centroid_points)
                if score and (score-np.sum(np.min(dist, axis=1)))<self.tol:
                    modelset.append((score,y_label,centroid_points))
                    break
                else:
                    score = np.sum(np.min(dist, axis=1))
                print('score:',score)
        print(modelset)
        return sorted(modelset, key=lambda x:x[0])[0]


