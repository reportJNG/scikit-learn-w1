'''Task 3 — Multiple Features

Now instead of using one input, we use many inputs.

Before:

X = [[4]]

Means:

4 study hours

Now:

X = [[4, 80]]

Means:

4 study hours + 80% attendance
Concept 🧠
One feature
X = [[1], [2], [3]]

Shape idea:

3 students
1 feature
Two features
X = [
    [1, 50],
    [2, 60],
    [3, 70],
]

Shape idea:

3 students
2 features

Each row = one student.

Each column = one information.

Task 3 — Predict Pass/Fail from 2 features
Data
X = [
    [1, 40],
    [2, 50],
    [3, 60],
    [4, 65],
    [5, 70],
    [6, 80],
    [7, 85],
    [8, 90],
]

y = [0, 0, 0, 0, 1, 1, 1, 1]

Meaning:

[study_hours, attendance_percentage]

Example:

[6, 80]

Means:

Student studied 6 hours and has 80% attendance
Your program must do ✅

Use LogisticRegression.

Predict:

[4.5, 75]

and:

[7, 90]

Also calculate accuracy.'''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# data
X = [
    [1, 40],
    [2, 50],
    [3, 60],
    [4, 65],
    [5, 70],
    [6, 80],
    [7, 85],
    [8, 90],
]

y = [0, 0, 0, 0, 1, 1, 1, 1]

X_train,X_test,y_train,y_test = train_test_split(

    X,y,test_size=0.3,random_state=42

)

#the model

model = LogisticRegression()

model.fit(X_train,y_train)


prediction1 = model.predict([[4.5, 75]])
prediction2 = model.predict([[7, 90]])
#i wanna make this comment cuz i actually did got it just now wow the logique were so easy and i was dumb so dumb to not get it when u try to test value u try with testing one cuz that the 30 % left in my logique not trying to test with value y train or x train cuz that value its only for training to model

y_pred = model.predict(X_test)
calc = accuracy_score(y_test,y_pred)

print('this is result for 4.5:',prediction1[0],'this is result for 6:',prediction2[0],'this is the accuarccy :',calc)