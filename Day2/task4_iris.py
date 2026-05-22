'''Program requirements:

1. Load iris dataset
2. Set X = iris.data
3. Set y = iris.target
4. Split train/test
5. Train LogisticRegression
6. Predict one flower using X_test[0]
7. Calculate accuracy
8. Print prediction, real answer, and accuracy
Starter only
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

# continue from here
Mini hints
iris.target_names

shows class names:

setosa
versicolor
virginica
'''



from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#data

iris = load_iris()

X=iris.data
y=iris.target

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

# model

model =LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
accuratty = accuracy_score(y_test,y_pred)

prediction = model.predict([X_test[0]])

print('accuraccy : ',accuratty)
print ('prediction : ',prediction[0] ,  'flower name : ',iris.target_names[prediction[0]])
print("Real answer:", y_test[0])
