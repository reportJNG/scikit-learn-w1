'''New concept: confusion matrix

Accuracy tells:

How many predictions were correct overall

Confusion matrix tells:

Where the model made mistakes
Task 5 — Iris Confusion Matrix 🌸

Create file:

task5_confusion_matrix.py
Goal

Use the Iris dataset again, but now print:

accuracy
confusion matrix
classification report
Required imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
Requirements

Your program must:

1. Load iris dataset
2. Set X = iris.data
3. Set y = iris.target
4. Split train/test
5. Train LogisticRegression
6. Predict X_test
7. Print accuracy
8. Print confusion matrix
9. Print classification report
10. Print target names'''

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#data
iris = load_iris()

X=iris.data
y=iris.target

X_train,X_test,y_train,y_test= train_test_split(
    X,y,test_size=0.3,random_state=42
)
#model
model = LogisticRegression()
model.fit(X_train,y_train)

predict = model.predict(X_test)

acuuraccy=accuracy_score(y_test,predict)
confusion = confusion_matrix(y_test,predict)
classfication=classification_report(y_test,predict)

print('Prediction :',predict)
print ('accuraccy :',acuuraccy)
print('confusion :',confusion)
print ('classification : ',classfication )
print('targetnames : ',iris.target_names[predict])