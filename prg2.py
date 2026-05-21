import numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(0); X=np.random.rand(100,1); y=2+3*X+np.random.randn(100,1)
m1=LinearRegression(fit_intercept=False).fit(X,y); m2=LinearRegression().fit(X,y)

plt.scatter(X,y,label='Data points')
plt.plot(X,m1.predict(X),color='red',label='Regression without bias')
plt.plot(X,m2.predict(X),color='blue',label='Regression with bias')
plt.legend(); plt.title('Linear Regression Model with and without Bias')
plt.xlabel('X'); plt.ylabel('y'); plt.show()

print("Model parameters without bias:")
print("Slope:",m1.coef_[0][0])
print("\nModel parameters with bias:")
print("Intercept:",m2.intercept_[0])
print("Slope:",m2.coef_[0][0])