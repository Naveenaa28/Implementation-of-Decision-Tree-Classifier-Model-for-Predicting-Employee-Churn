# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics.

10.Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:NAVEENAA V.R 
RegisterNumber:212221220035  
*/
```
```

import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
## Data head:
![image](https://github.com/Naveenaa28/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131433133/daddb771-8a91-4f04-b78d-fb42e3fa6a6d)
## Dataset info:
![image](https://github.com/Naveenaa28/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131433133/5656f02f-e05b-4b88-ad62-fcc99fb61919)
## Null dataset:
![image](https://github.com/Naveenaa28/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131433133/896178c4-5b16-4bfa-b459-01bddbb16bd1)
## Values count in left column:
![image](https://github.com/Naveenaa28/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131433133/34cdf0e4-cf49-41b6-bedb-6b0a08ec0a4a)
## Dataset transformed head:
![image](https://github.com/Naveenaa28/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131433133/38b57390-787b-44f5-98f7-27271915cb99)
## x.head
![image](https://github.com/Naveenaa28/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131433133/df86a46e-82e3-4119-93ea-d1a82242a56c)
## Accuracy
![image](https://github.com/Naveenaa28/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131433133/a8ae95be-37ec-4aa4-82e9-320861ea8ce8)
## Data Prediction
![image](https://github.com/Naveenaa28/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131433133/eb1f47f5-4157-482f-b979-19706eb8caa1)
## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
