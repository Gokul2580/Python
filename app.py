import numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X=np.random.rand(100,1); y=2+3*X+np.random.randn(100,1)
plt.scatter(X,y); plt.title("Scatter plot of X vs y"); plt.xlabel("X"); plt.ylabel("y"); plt.show()

print("Correlation coefficient between X and y:",np.corrcoef(X[:,0],y[:,0])[0,1])
model=LinearRegression().fit(X,y); print("Intercept:",model.intercept_[0]); print("Slope:",model.coef_[0][0])

plt.scatter(X,y); plt.plot(X,model.predict(X),color='red')
plt.title("Linear regression model"); plt.xlabel("X"); plt.ylabel("y"); plt.show()