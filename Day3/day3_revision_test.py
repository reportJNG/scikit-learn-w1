'''Day 3 — Mini Revision Test

Create CSV:

day3_revision.csv

Put this inside:

age,income,visits,total_spent,bought
18,20000,1,50,0
22,25000,2,80,0
25,30000,3,120,0
30,40000,4,200,1
35,50000,5,350,1
40,60000,6,500,1
45,70000,7,700,1
50,80000,8,900,1
28,,3,150,0
33,45000,,300,1
Your task

Create:

day3_revision_test.py
Part A — Cleaning

Do:

1. Read CSV with pandas
2. Print df.head()
3. Print missing values before cleaning
4. Fill missing values using df.mean()
5. Print missing values after cleaning
Part B — Regression

Predict:

total_spent

Features:

age
income
visits

Use:

LinearRegression

Predict this customer:

[[32, 47000, 4]]
Part C — Classification

Predict:

bought

Features:

age
income
visits

Use:

LogisticRegression

Predict:

[[32, 47000, 4]]

Also calculate:

accuracy
confusion matrix
classification report'''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
)
import pandas as pd

data = pd.read_csv("./CSV/day3_revision.csv")

# Cleaning
print("Missing before cleaning:")
print(data.isnull().sum())

data = data.fillna(data.mean())

print("Missing after cleaning:")
print(data.isnull().sum())

print(data.head())

# Features
X = data[["age", "income", "visits"]]

# -------------------------
# Part B: Regression
# -------------------------
y_score = data["total_spent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_score, test_size=0.3, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict([[32, 47000, 4]])

y_pred = model.predict(X_test)
error = mean_absolute_error(y_test, y_pred)

print("Predicted total spent:", prediction[0])
print("Mean absolute error:", error)

# -------------------------
# Part C: Classification
# -------------------------
y_bought = data["bought"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X, y_bought, test_size=0.3, random_state=42
)

model2 = LogisticRegression()
model2.fit(X2_train, y2_train)

prediction2 = model2.predict([[32, 47000, 4]])

y_pred2 = model2.predict(X2_test)

accuracy2 = accuracy_score(y2_test, y_pred2)
report2 = classification_report(y2_test, y_pred2)
confusion2 = confusion_matrix(y2_test, y_pred2)

print("Bought prediction:", prediction2[0])
print("Accuracy:", accuracy2)
print("Report:")
print(report2)
print("Confusion matrix:")
print(confusion2)