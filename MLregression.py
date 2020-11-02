import pandas
df = pandas.read_csv("Advertising.csv")
print(df)

# in regression we predict numeric, continous variables i.e sales, weight, population, price, sales, speed, forex, score

import sklearn
array = df.values
X=array[:,1:4]
y=array[:, 4]

from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.30, random_state=42)

#train data
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# ask model to predict x_test, hide y_test(answers)
predictions = model.predict(X_test)
print(predictions)

from sklearn.metrics import r2_score
print('R squared', r2_score(y_test, predictions))

import matplotlib.pyplot as plot
# change style of graph
plot.style.use('seaborn')
figure, ax = plot.subplots()
ax.scatter(y_test, predictions)
ax.plot(y_test, y_test)
ax.set_title('y_test vs. predictions')
ax.set_xlabel('Y-test')
ax.set_ylabel('predictions')
plot.show()

expenditure = [[142.9,29.3,12.6]]
sales = model.predict(expenditure)
print(sales)
