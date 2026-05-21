from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [30000, 35000, 40000, 45000, 50000, 60000, 65000, 70000]

X_train,X_test,Y_train,Y_test = train_test_split (
    X,y,test_size=0.25,random_state=12
)

model=LinearRegression()

model.fit(X_train,Y_train)

predict = model.predict([[9]])

print('predict a salary based on the year' ,predict[0] )