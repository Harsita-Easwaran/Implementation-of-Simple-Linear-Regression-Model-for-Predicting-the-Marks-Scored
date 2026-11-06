# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Harsita Easwaran
RegisterNumber:  212224220036
*/
```

```
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_absolute_error, mean_squared_error 
df=pd.read_csv('student_scores.csv')
print(df)
df. head (0) 
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print (x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt. title("Hours vs Scores(Training set)")
plt.xlabel ("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel ("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ' ,mse)
mae=mean_absolute_error(y_test,y_pred)
print( 'MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
<img width="396" height="1130" alt="image" src="https://github.com/user-attachments/assets/fd83cee4-5253-4bc3-8973-8ab8a4ff91c0" />
<img width="1146" height="1252" alt="image" src="https://github.com/user-attachments/assets/b13aaeb8-bb33-457c-b92d-b2e254ba739d" />
<img width="1122" height="902" alt="image" src="https://github.com/user-attachments/assets/58ce2eed-341f-460e-b7a0-685d26a28780" />
<img width="1124" height="1044" alt="image" src="https://github.com/user-attachments/assets/eead263a-d2b4-41ad-acfd-e8f8988ed240" />




## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
