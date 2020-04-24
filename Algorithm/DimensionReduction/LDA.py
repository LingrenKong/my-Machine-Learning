# _*_coding:utf-8_*_
"""
LDA 线性判别分析
Author: Lingren Kong
Created Time: 2020/4/22 21:58
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

class LDA_simple():
    import numpy as np
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.means_ =None
        print("单一功能LDA线性判别类")

    def fit(self,X,y):
        """

        Parameters
        ----------
        X :输入数据m*d
        y :类别标签m*1 0-1标签

        Returns 修改参数w
        -------

        """
        X0 = X[y==0,:]
        X1 = X[y==1,:]
        mu0 = np.mean(X0, axis=0)
        mu1 = np.mean(X1, axis=0)
        self.means_ = np.array((mu0,mu1))
        Sw = np.dot((X0-mu0).T,X0-mu0) + np.dot((X1-mu1).T,X1-mu1)
        U, Sigma, Vt = np.linalg.svd(Sw)
        Sw_i = Vt.T * np.linalg.inv(np.diag(Sigma)) * U.T
        self.coef_ = np.dot(Sw_i,(mu0-mu1).reshape(mu0.shape[0],1))#是n*1形式的参数
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_))

    def transform(self,X):
        return np.dot(X,self.coef_)





if '__main__' == __name__:
    X, y = make_classification(n_samples=300, n_features=2,n_classes=2,n_redundant=0,
                               n_informative=1, n_clusters_per_class=1, random_state=2333)
    #制造一份数据
    lda = LDA_simple()
    lda.fit(X,y)
    print('我的模型--参数：',lda.coef_,'阈值',lda.intercept_)
    plt.scatter(X[:, 0], X[:, 1], marker='+', c=y)
    plt.show()
    projection = lda.transform(X)
    projection = projection.reshape(projection.shape[0])
    plt.scatter(projection,y,c=y)
    plt.show()
    print('-----')
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    print('sklearn模型--系数', clf.coef_ / np.sqrt(np.sum(clf.coef_ ** 2)), '阈值', clf.intercept_)
    projection = clf.transform(X)
    projection = projection.reshape(projection.shape[0])
    plt.scatter(projection, y, c=y)
