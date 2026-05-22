'''Day 2 — Task 4: Employee Promotion Classifier
Goal

Predict if an employee will be promoted or not.

0 = not promoted
1 = promoted

Use:

LogisticRegression
Step 1 — Create CSV file

Create:

employee_promotion.csv

Put this inside:

work_hours,projects_done,performance_score,promoted
35,1,45,0
38,2,50,0
40,2,55,0
42,3,58,0
45,3,62,0
48,4,68,1
50,4,72,1
52,5,75,1
55,5,80,1
58,6,85,1
60,7,90,1
44,2,60,0
49,4,70,1
53,5,78,1
Step 2 — Create Python file
task7_employee_promotion.py
Your task ✅

Your program must:

1. Import pandas
2. Read employee_promotion.csv
3. Print df.head()
4. Print df.columns
5. Set X = work_hours, projects_done, performance_score
6. Set y = promoted
7. Split train/test
8. Train LogisticRegression
9. Predict this employee: [51, 4, 74]
10. Predict this employee: [39, 2, 52]
11. Predict X_test
12. Calculate accuracy
13. Print custom predictions + test predictions + accuracy'''

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

data  = pd.read_csv('./csv/employee_promotion.csv')

X=data[['work_hours','projects_done','performance_score']]
y=data['promoted']

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

#model
model = LogisticRegression()
model.fit(X_train,y_train)
predict1=model.predict([[51, 4, 74]])
predict2=model.predict([[39, 2, 52]])

y_pred=model.predict(X_test)
accuratte=accuracy_score(y_test,y_pred)

#priting data
print('predict 1 :',predict1)
print('predict 2 :',predict2)
print('model predicts : ',y_pred)
print('accuracy',accuratte)