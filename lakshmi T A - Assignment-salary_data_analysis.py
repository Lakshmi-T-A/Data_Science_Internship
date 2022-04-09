from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset

data = pd.read_csv('Salary_Data.csv')

#print(data)

x = data.iloc[:, :-1].values
y = data.iloc[:,1].values

#splitting dataset

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 1/3, random_state = 0)

print(x_test)

#fitting the model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
print(y_pred)

#visualizing the results

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()