# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas
2.Import Decision tree classifier
3.Fit the data in the model
4.Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: John Wilfred Thomas J W
RegisterNumber:  24013517
*/
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics


data = pd.read_csv("Salary.csv")


print(data.head())
print(data.info())
print(data.isnull().sum())


le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])


x = data[["Position", "Level"]]
y = data["Salary"]              


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)


y_pred = dt.predict(x_test)


r2 = metrics.r2_score(y_test, y_pred)
print(f"R-squared: {r2}")
print("Predicted salaries:", y_pred)


## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
![image](https://github.com/user-attachments/assets/b71029c9-be04-49ed-a2e9-d3b773a9da5a)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
