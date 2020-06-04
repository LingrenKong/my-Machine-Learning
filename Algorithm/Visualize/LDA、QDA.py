# _*_coding:utf-8_*_
"""
Author: Lingren Kong
Created Time: 2020/4/25 10:25
"""

"""
对于非均等协方差情况下的LDA与QDA可视化与模型评估
"""
import numpy as np

def blob_plus(mean, cov, n_samples_foreach=100, label=None):
    """

    Parameters
    ----------
    mean:k*d数组，k为类别数，d为数据维数
    cov:k*d*d数组，k个协方差矩阵
    label:各个类别的标签，默认0~k-1

    Returns
    -------

    """
    import numpy as np
    k = mean.shape[0]
    d = mean.shape[1]
    if not label:
        label = list(range(k))
    X = np.empty((0,d))
    y = np.empty(0)
    for i in range(k):
        X = np.append(X,np.random.multivariate_normal(mean[i,:],cov[i,:,:],n_samples_foreach),axis=0)
        y = np.append(y,i*np.ones(n_samples_foreach),axis=0)

    return X,y

# 随机生成不同协方差的高斯分布
np.random.seed(2)
mean = np.array([[3, 6],[3, -2]])
cov = np.array([[[0.5, 0], [0, 2]],[[2, 0], [0, 2]]])
X,y = blob_plus(mean,cov,1000)
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test =train_test_split(X,y,test_size=0.3)
x0 = np.linspace(-2,8,num=1000)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train,y_train)
y1 = (-lda.coef_[0,0]*x0-lda.intercept_)/lda.coef_[0,1]
print(lda.coef_,lda.intercept_)
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train,y_train)
y2 = 3.514-1.125*x0+0.1875*x0**2

import matplotlib.pyplot as plt
# 样本点
plt.scatter(X[:,0],X[:,1],c=y,marker='+')
# 似然估计的数据中心
plt.plot(lda.means_[0][0], lda.means_[0][1],
         '*', color='yellow', markersize=15, markeredgecolor='grey')
plt.plot(lda.means_[1][0], lda.means_[1][1],
         '*', color='gray', markersize=15, markeredgecolor='grey')
# 画两个决策界面
plt.plot(x0,y1)
plt.plot(x0,y2)
plt.text(-2,8,f'LDA SCORE:{lda.score(X_test,y_test):.4}')
plt.text(6,0,f'QDA SCORE:{qda.score(X_test,y_test):.4}')
plt.show()