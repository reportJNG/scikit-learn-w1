'''Great. Day 3 — Task 3 🔥
New skill: predict a custom customer after get_dummies().

This is important because after encoding, your columns change.

Problem

Before encoding:

age, income, city, plan, visits

After pd.get_dummies():

age
income
visits
city_Alger
city_Blida
city_Oran
plan_basic
plan_premium

So when you predict a new customer, you must give the model same columns as X.

Day 3 — Task 3: Custom Customer Prediction

Use the same CSV from previous task:

day3_task2_marketing.csv
Your program must do

Create:

day3_task3_custom_customer.py

Requirements:

1. Read CSV
2. Fill missing numeric values
3. Encode city and plan using pd.get_dummies()
4. Set X = all columns except bought
5. Set y = bought
6. Split train/test
7. Train LogisticRegression
8. Create custom customer manually
9. Make sure custom customer has same columns as X
10. Predict custom customer
11. Predict X_test
12. Calculate accuracy
Custom customer to predict

Customer:

age = 32
income = 48000
visits = 4
city = Alger
plan = premium

After encoding, it should look like:

custom_customer = pd.DataFrame([{
    "age": 32,
    "income": 48000,
    "visits": 4,
    "city_Alger": 1,
    "city_Blida": 0,
    "city_Oran": 0,
    "plan_basic": 0,
    "plan_premium": 1,
}])
Important line 🧠

After creating custom customer, use this:

custom_customer = custom_customer[X.columns]

Meaning:

Reorder custom customer columns to match X exactly.

This prevents column order mistakes.'''

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('./CSV/day3_task2_marketing.csv')
row = df.select_dtypes(include='number').columns
df[row]=df[row].fillna(df[row].mean())
df=pd.get_dummies(df,columns=['city','plan'],dtype=int)

X=df.drop('bought',axis=1)
y=df['bought']

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

#model

model=LogisticRegression(max_iter=200)
model.fit(X_train,y_train)

custom_customer = pd.DataFrame([{
    "age": 32,
    "income": 48000,
    "visits": 4,
    "city_Alger": 1,
    "city_Blida": 0,
    "city_Oran": 0,
    "plan_basic": 0,
    "plan_premium": 1,
}])
custom_customer = custom_customer[X.columns]
customprediction = model.predict(custom_customer)
print(customprediction)
y_pred = model.predict(X_test)
accurate = accuracy_score(y_test,y_pred)
print(accurate)