# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Harsita Easwaran
RegisterNumber:  24013629
*/
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

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
  Hours  Scores
0    2.5      21
1    5.1      47
2    3.2      27
3    8.5      75
4    3.5      30
    Hours  Scores
20    2.7      30
21    4.8      54
22    3.8      35
23    6.9      76
24    7.8      86
[[2.5]
 [5.1]
 [3.2]
 [8.5]
 [3.5]
 [1.5]
 [9.2]
 [5.5]
 [8.3]
 [2.7]
 [7.7]
 [5.9]
 [4.5]
 [3.3]
 [1.1]
 [8.9]
 [2.5]
 [1.9]
 [6.1]
 [7.4]
 [2.7]
 [4.8]
 [3.8]
 [6.9]
 [7.8]]
[21 47 27 75 30 20 88 60 81 25 85 62 41 42 17 95 30 24 67 69 30 54 35 76
 86]
[17.04289179 33.51695377 74.21757747 26.73351648 59.68164043 39.33132858
 20.91914167 78.09382734 69.37226512]
[20 27 69 30 62 35 24 86 76]

![Screenshot 2024-10-29 143922](https://github.com/user-attachments/assets/96241f11-bc68-44e5-83e0-f88f8303c49d)

![Screenshot 2024-10-29 143951](https://github.com/user-attachments/assets/ddcb4e1e-bd7f-48ae-ad11-20561d4d1d18)

MSE =  4.691397441397438
MAE =  4.691397441397438
RMSE=  2.1659633979819324



## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
