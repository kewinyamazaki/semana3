import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = read.csv(r'C:\Users\Pichau\Downloads\dados.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,:-1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15,random_state=0)

from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

y_pred = linear_regression.fit(X_train, y_train)

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, linear_regression.predict(X_train), color='blue')
plt.title("Salário x Tempo de Experiência (Treinamento)")
plt.xlabel("Anos de experiência")
plt.ylabel("Salário")
plt.show()

print(f'y={linear_regression.coef_[0]:.2f}x+{linear_regression.intercept_:.2f}')

