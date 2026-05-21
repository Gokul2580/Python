import pandas as pd; from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron; from sklearn.metrics import accuracy_score

d=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
names=['sl','sw','pl','pw','s'])

X_train,X_test,y_train,y_test=train_test_split(d.drop('s',axis=1),d['s'],test_size=0.2,random_state=0)

m1=Perceptron(fit_intercept=False).fit(X_train,y_train); print("Accuracy without bias:",accuracy_score(y_test,m1.predict(X_test)))

m2=Perceptron().fit(X_train,y_train); print("Accuracy with bias:",accuracy_score(y_test,m2.predict(X_test)))