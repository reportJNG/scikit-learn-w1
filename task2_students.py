from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#data
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [0, 0, 0, 0, 1, 1, 1, 1]

#model

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

model = LogisticRegression()
model.fit(X_train,y_train)

prediction1 = model.predict([[4.5]])
prediction2 = model.predict([[6]])
# X_test = test inputs
# y_test = real answers for those test inputs
# y_pred = model predictions for X_test
# accuracy compares real answers vs predicted answers

y_pred = model.predict(X_test)
calc= accuracy_score(y_test,y_pred)


print('this is result for 4.5:',prediction1[0],'this is result for 6:',prediction2[0],'this is the accuarccy :',calc)