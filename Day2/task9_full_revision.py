'''Task 6 — Student ML Full Revision

Create CSV file:

student_revision.csv

Put this inside:

study_hours,attendance,previous_score,final_score,passed
1,40,25,28,0
2,45,30,35,0
3,55,40,45,0
4,60,48,52,0
5,70,60,65,1
6,75,68,72,1
7,85,80,84,1
8,90,88,92,1
4,,50,55,0
6,80,,74,1

Notice: there are missing values.

Your program must do

Create:

task6_full_revision.py
Part A — Pandas cleaning

Do this first:

1. Import pandas
2. Read student_revision.csv
3. Print df.head()
4. Print missing values before cleaning
5. Fill missing values using df.mean()
6. Print missing values after cleaning
Part B — Regression

Goal: predict final_score.

Use:

LinearRegression

Features:

study_hours
attendance
previous_score

Target:

final_score

Your code must:

1. Set X
2. Set y_score
3. Split train/test
4. Train LinearRegression
5. Predict final score for [6.5, 78, 70]
6. Print prediction
Part C — Classification

Goal: predict passed.

Use:

LogisticRegression

Features:

study_hours
attendance
previous_score

Target:

passed

Your code must:

1. Set X
2. Set y_pass
3. Split train/test
4. Train LogisticRegression
5. Predict pass/fail for [6.5, 78, 70]
6. Predict pass/fail for [3, 55, 40]
7. Predict X_test
8. Calculate accuracy
9. Print confusion matrix
10. Print classification report'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pandas as panda

#fixing full data of the csv file 

data = panda.read_csv('./csv/student_revision.csv')

#print (data.isnull.sum())
#print(data.head())

data = data.fillna(data.mean())

#print (data.isnull().sum())

##train data
X=data[['study_hours','attendance','previous_score']]
y1=data['final_score']

X1_train,X1_test,y1_train,y1_test = train_test_split(
    X,y1,test_size=0.3,random_state=42
)

#model

model = LinearRegression()
model.fit(X1_train,y1_train)

prediction1 = model.predict([[6.5, 78, 70]])

#print 
print('prediction 1 : ',prediction1)


#traning 2 :
y2=data['passed']

X2_train,X2_test,y2_train,y2_test = train_test_split(
    X,y2,test_size=0.3,random_state=42
)

#model 2 

model2 = LogisticRegression()

model2.fit(X2_train,y2_train)

predict2 =model2.predict([[6.5, 78, 70]])
predict3 =model2.predict([[3, 55, 40]])

y_pred = model2.predict(X2_test)

accurate = accuracy_score(y2_test,y_pred)

matrix = confusion_matrix(y2_test,y_pred)
report = classification_report(y2_test,y_pred)

#print confusion + matrix
print(matrix)
print (report)