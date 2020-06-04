# _*_coding:utf-8_*_
"""
单隐层BP神经网络，激活函数都采用sigmoid；模型中的参数名称参照西瓜书BP
Author: Lingren Kong
Created Time: 2020/6/4 21:33
"""
import numpy as np
class BP():

    def __init__(self, hidden, fit='standard', randomState = 0, tol = 1e-6, maxiter=200,eta=0.1):
        """

        Parameters
        ----------
        hidden : 隐藏层单元数
        randomState：随机数种子
        maxiter:最大迭代数
        eta:学习率
        theta和gamma是阈值；w和v是权值
        tol迭代终止条件
        """
        self.hidden = hidden
        self.randomState = randomState
        np.random.seed(randomState)
        self.theta = None
        self.gamma = None
        self.w = None
        self.v = None
        self.pre_loss = None
        self.fitmethod = fit
        self.maxiter = maxiter
        self.eta = eta
        self.tol = tol
        if fit not in ['standard','accumulate']:
            print("方法有误")
            self.fit = 'standard'

    def sigmoid(self,x):
        """

        Parameters
        ----------
        x 输入数据（标量）

        Returns sigmoid
        -------

        """
        return 1.0 / (1.0 + np.exp(-x))

    def randParam(self, X, y):
        """

        Parameters
        ----------
        X
        y

        Returns 初始化一下参数
        -------

        """
        try:
            ylen = y.shape[1]
        except:
            ylen=1
        self.theta = np.random.random(ylen)
        self.gamma = np.random.random(self.hidden)
        self.w = np.random.random((self.hidden,ylen))
        self.v = np.random.random((X.shape[1],self.hidden))

    def forward(self,X):
        self.b = self.sigmoid(np.dot(X.reshape(1,-1),self.v)-self.gamma).reshape(-1)
        y_pred = self.sigmoid(np.dot(self.b,self.w)-self.theta)
        #print("y_pred:",y_pred)
        return y_pred

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])#暂时按照只有1个
        for i in range(X.shape[0]):
            y_pred[i] = self.forward(X[i,:])
        return y_pred

    def loss(self,X,y):
        return np.sum((self.predict(X)-y)**2)/2

    def fit_standard(self, X, y):
        self.randParam(X, y)
        terminate = 0
        for _ in range(self.maxiter):
            for i in range(X.shape[0]):
                x0 = X[i,:]
                try:
                    y0 = y[i,:]
                except:
                    y0 = y[i]
                y_pred = self.forward(x0)
                g = y_pred*(1-y_pred)*(y0-y_pred)
                #temp = np.dot(self.w, g.reshape(-1, 1))
                e = self.b * (1 - self.b) * np.dot(self.w, g.reshape(-1, 1)).reshape(-1)
                self.w += self.eta*np.dot(self.b.reshape(-1, 1), g.reshape(1, -1))#w_hj = eta*b_h*g_J 转换矩阵表达式
                self.theta += -self.eta*g
                self.v += self.eta*np.dot(x0.reshape(-1,1),e.reshape(1,-1))
                self.gamma += -self.eta*e
                #print(self.pre_loss,self.loss(X,y))
                '''
                if self.pre_loss and self.pre_loss-self.loss(X,y)<self.tol:
                    terminate += 1
                    if terminate>X.shape[0]:#因为标准BP的随机性，所以终止门槛要高一些
                        print(f'在{_*X.shape[0]+i}次提前跳出')
                        return
                else:
                    terminate = 0
                    self.pre_loss = self.loss(X,y)
                '''

        print(f"最终的损失函数为{self.loss(X, y)}")

    def fit_accumulate(self, X, y):
        self.randParam(X, y)
        for _ in range(self.maxiter):
            dw = 0
            dtheta = 0
            dv = 0
            dgamma = 0
            for i in range(X.shape[0]):
                x0 = X[i,:]
                try:
                    y0 = y[i,:]
                except:
                    y0 = y[i]
                y_pred = self.forward(x0)
                g = y_pred*(1-y_pred)*(y0-y_pred)
                #temp = np.dot(self.w, g.reshape(-1, 1))
                e = self.b * (1 - self.b) * np.dot(self.w, g.reshape(-1, 1)).reshape(-1)
                dw += self.eta*np.dot(self.b.reshape(-1, 1), g.reshape(1, -1))#w_hj = eta*b_h*g_J 转换矩阵表达式
                dtheta += -self.eta*g
                dv += self.eta*np.dot(x0.reshape(-1,1),e.reshape(1,-1))
                dgamma += -self.eta*e
            self.w += dw
            self.theta += dtheta
            self.v += dv
            self.gamma += dgamma

            '''
            t =  self.loss(X, y)
            if self.pre_loss and self.pre_loss - t < self.tol:
                print(f'在{_}次提前跳出')
                return
            else:
                self.pre_loss = t
            '''
        print(f"最终的损失函数为{self.loss(X, y)}")

    def fit(self,X,y):
        if self.fitmethod == 'standard':
            self.fit_standard(X,y)
        else:
            self.fit_accumulate(X, y)

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral,levels=1)
    plt.colorbar()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


if __name__ == '__main__':
    import sklearn.datasets
    import matplotlib.pyplot as plt
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    #plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    #plt.show()
    model1 = BP(50,maxiter=200,fit='accumulate')#3个结点和线性没有区别
    model1.fit(X,y)
    plot_decision_boundary(model1.predict)

    model2 = BP(10, maxiter=200)
    model2.fit(X,y)
    plot_decision_boundary(model2.predict)
