from myML.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 2333
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

X2,y2 = make_blobs(n_samples=20, random_state=random_state)
obj = KMeans(n_clusters=3,n_init=10, max_iter=300)
result = obj.fit(X)
plt.scatter(X[:, 0], X[:, 1], c=result[1])
plt.show()

#plt.scatter(X[:, 0], X[:, 1])
#plt.title("emm")
#plt.show()
