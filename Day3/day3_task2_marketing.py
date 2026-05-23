'''Create:

day3_task2_marketing.py

Your program must:

1. Read CSV
2. Print missing values before cleaning
3. Fill missing numeric values using mean
4. Print missing values after cleaning
5. Convert city and plan using pd.get_dummies()
6. Set X = all columns except bought
7. Set y = bought
8. Split train/test
9. Train LogisticRegression
10. Predict first test customer using X_test.iloc[[0]]
11. Predict all X_test
12. Calculate accuracy
13. Print prediction + accuracy'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as panda

data = panda.read_csv('./CSV/day3_task2_marketing.csv')
#print(data.isnull().sum())
calcarow= data.select_dtypes(include=['number']).columns
#i dont think there is missing num in data but this is how u do fix data in pandas csv by calculation tfirst the row [i] 
data[calcarow] = data[calcarow].fillna(data[calcarow].mean())
data = panda.get_dummies(data,columns=['city','plan'],dtype=int)

X=data.drop('bought',axis=1)
y=data['bought']

#model

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train,y_train)
predict1 = model.predict(X_test.iloc[[0]])
predictall=model.predict(X_test)
print(predictall)
accurate=accuracy_score(y_test,predictall)
print(accurate)