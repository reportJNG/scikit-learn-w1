'''Day 4 — Quick Revision Test

Create CSV:

day4_revision.csv

Put this:

age,income,visits,city,plan,total_spent,bought
18,20000,1,Alger,basic,50,0
22,25000,2,Oran,basic,80,0
25,30000,3,Blida,basic,120,0
30,40000,4,Alger,premium,200,1
35,50000,5,Blida,premium,350,1
40,60000,6,Oran,premium,500,1
45,70000,7,Alger,premium,700,1
50,80000,8,Oran,premium,900,1
28,,3,Blida,basic,150,0
33,45000,,Alger,premium,300,1
Your task

Create:

day4_revision_test.py
Requirements
Part A — Cleaning + Encoding
1. Read CSV
2. Print missing values before cleaning
3. Fill missing numeric values using mean
4. Print missing values after cleaning
5. Convert city and plan using pd.get_dummies()
6. Print encoded columns
Part B — Regression

Predict:

total_spent

Features:

all columns except total_spent and bought

Use:

LinearRegression

Evaluate with:

mean_absolute_error

Predict this custom row manually after encoding:

age = 32
income = 47000
visits = 4
city = Alger
plan = premium

For this revision test, you can manually encode the custom row.

Part C — Classification

Predict:

bought

Features:

all columns except total_spent and bought

Use:

LogisticRegression

Evaluate with:

accuracy_score
confusion_matrix
classification_report

Predict the same custom customer.'''

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score,mean_absolute_error,confusion_matrix,classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression

data=pd.read_csv('./CSV/day4_revision.csv')
#print(data.isnull().sum())

#first we can't clean data with knowing the columns number so

row = data.select_dtypes(include=['number']).columns
#print(row)
#now we can start cleaning the actuall data

data[row] = data[row].fillna(data[row].mean())
#print(data)
data = pd.get_dummies(data,['city','plan'],dtype=int)
#print(data)
#now after cleaning full data we can put then into input and target to train our model

X=data.drop(['total_spent','bought'],axis=1)
y=data['total_spent']

#start to train the first model
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)
model1=LinearRegression()
model1.fit(X_train,y_train)
custom = pd.DataFrame([{
    'age': 32,
    'income': 47000,
    'visits': 4,
    'city': 'Alger',
    'plan': 'premium'
}])
#adding custum data that our model isnt trained at will be needed to fixed too
custom = pd.get_dummies(custom, ['city', 'plan'], dtype=int)
custom = custom.reindex(columns=X.columns, fill_value=0)

prediction = model1.predict(custom)
#print(prediction[0])
y_pred = model1.predict(X_test)
mm=mean_absolute_error(y_test,y_pred)
#print(mm)
#-------------------------------------------------------------------
X1=data.drop(['bought','total_spent'],axis=1)
y1=data['bought']

X1_test,X1_train,y1_test,y1_train = train_test_split(
    X1,y1,test_size=0.3,random_state=42
)

model2=LogisticRegression(max_iter=200)
model2.fit(X1_train,y1_train)
y_pred2=model2.predict(X1_test)
#print(predict2[0])
print(classification_report(y1_test,y_pred2))
print(accuracy_score(y1_test,y_pred2))
print(confusion_matrix(y1_test,y_pred2))
