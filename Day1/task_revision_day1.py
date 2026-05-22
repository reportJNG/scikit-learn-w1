'''Part A — Regression
Goal

Predict the final exam score using:

study hours
attendance %

Because final score is a number, use:

LinearRegression
Data
X = [
    [1, 40],
    [2, 50],
    [3, 55],
    [4, 60],
    [5, 70],
    [6, 75],
    [7, 85],
    [8, 90],
]

y_score = [25, 35, 40, 50, 60, 68, 80, 90]
Required

Your code must:

1. Split data into train/test
2. Train LinearRegression
3. Predict score for [6.5, 80]
4. Print the predicted score
Part B — Classification
Goal

Predict if the student will pass or fail.

0 = fail
1 = pass

Because the result is a class, use:

LogisticRegression
Data

Use the same X, but new target:

y_pass = [0, 0, 0, 0, 1, 1, 1, 1]
Required

Your code must:

1. Split data into train/test
2. Train LogisticRegression
3. Predict pass/fail for [6.5, 80]
4. Predict pass/fail for [3, 55]
5. Calculate accuracy using y_test and y_pred
6. Print predictions and accuracy'''


#imports
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#data A

X = [
    [1, 40],
    [2, 50],
    [3, 55],
    [4, 60],
    [5, 70],
    [6, 75],
    [7, 85],
    [8, 90],
]

y_score = [25, 35, 40, 50, 60, 68, 80, 90]

X1_train,X1_test,y1_train,y1_test = train_test_split(
    X,y_score,test_size=0.3,random_state=42)

#model A

modelA = LinearRegression()
modelA.fit(X1_train,y1_train)


predictA = modelA.predict([[6.5, 80]])

print('Prediction for Final exam score :',predictA[0])

#data B
y_pass = [0, 0, 0, 0, 1, 1, 1, 1]

X2_train,X2_test,y2_train,y2_test = train_test_split (
    X,y_pass , test_size=0.3 ,random_state=42
)

#model B
modelB=LogisticRegression()
modelB.fit(X2_train,y2_train)

predictionB1 = modelB.predict([[6.5, 80]])  
predictionB2 = modelB.predict([[3, 55]])

y_pred = modelB.predict(X2_test)
ispass = accuracy_score(y2_test,y_pred)

print ('Student one :',predictionB1,'Student two : ',predictionB2,'accuraty score : ',ispass)