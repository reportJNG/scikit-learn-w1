'''Your task

Create:

day3_task1_categories.py

Your program must:

1. Read CSV
2. Print df before encoding
3. Use pd.get_dummies() on city and plan
4. Print df after encoding
5. Set X = all columns except bought
6. Set y = bought
7. Split train/test
8. Train LogisticRegression
9. Predict one customer
10. Calculate accuracy'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as panda


data = panda.read_csv('./CSV/customer_category.csv')
data = panda.get_dummies(data,columns=['city','plan'])
print(data)
X=data.drop('bought',axis=1)
y=data['bought']

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

#model training 
model = LogisticRegression(max_iter=200)
model.fit(X_train,y_train)
first_customer = X_test[0].iloc[[0]]
predict = model.predict(first_customer)
y_pred = model.predict(X_test)
accurate = accuracy_score(y_test,y_pred)

#priting
print('prediction :',predict[0])
print ('accurate :',accurate)
