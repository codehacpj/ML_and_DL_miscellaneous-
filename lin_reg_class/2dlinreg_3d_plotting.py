import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
X=[]
Y=[]

for line in open('data_2d.csv'):
    data = line.split(',')
    X.append([float(data[0]),float(data[1]),1])
    Y.append(float(data[2]))
    
X = np.array(X)
Y = np.array(Y)

W = np.linalg.solve((X.T).dot(X),(X.T).dot(Y))
Y_pred = X.dot(W)

# ploting 
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0],X[:,1],Y)
ax.scatter(X[:,0],X[:,1],Y_pred,color='red')
plt.show()
