import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
X=[]
Y=[]

for line in open('data_poly.csv'):
    x,y = line.split(',')
    x = float(x)
    y = float(y)
    X.append([1,x,x*x])
    Y.append(y)
    
X = np.array(X)
Y = np.array(Y)


# calculating the weights
W = np.linalg.solve((X.T).dot(X),(X.T).dot(Y))
Y_pred = X.dot(W)

# plot 
plt.scatter(X[:,1],Y,color = 'red')
plt.plot(sorted(X[:,1]),sorted(Y_pred),color='green')

# R square value for above model
SSres = (Y - Y_pred).dot(Y - Y_pred)
SStot = (Y - Y.mean()).dot(Y - Y.mean())
R_sq = 1 - SSres/SStot
print("The R sqare value for polynomial regresion is: "+str(R_sq))
