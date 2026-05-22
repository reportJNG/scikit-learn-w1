'''Goal

Predict if a customer will buy a product.

0 = no
1 = yes

Use:

LogisticRegression
Step 1 — Create CSV file

Create:

customer_purchase.csv

Put this inside:

age,income,visits,bought
18,20000,1,0
22,25000,2,0
25,30000,3,0
30,40000,4,1
35,50000,5,1
40,60000,6,1
45,70000,7,1
50,80000,8,1
28,,3,0
33,45000,,1

Notice: there are empty values.

Your task ✅

Create:

task8_customer_purchase.py

Your program must:

1. Import pandas
2. Read customer_purchase.csv
3. Print df.head()
4. Print missing values using df.isnull().sum()
5. Fill missing values using column mean
6. Set X = age, income, visits
7. Set y = bought
8. Split train/test
9. Train LogisticRegression
10. Predict this customer: [32, 47000, 4]
11. Predict X_test
12. Calculate accuracy
13. Print prediction + accuracy'''

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('./csv/customer_purchase.csv')



#heere we will be needing to fill up the emptey output in csv
print (data.isnull().sum())
data = data.fillna(data.mean())
X=data[['age','income','visits']]
y=data['bought']
print (data.head())
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3 ,random_state=42
)

#model
model=LogisticRegression(max_iter=200)
model.fit(X_train,y_train)
predict = model.predict([[32, 47000, 4]])
y_pred=model.predict(X_test)
accuraccy=accuracy_score(y_test,y_pred)
print('prediction :',predict[0])
print('accurate :',accuraccy)