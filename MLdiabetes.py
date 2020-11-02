import pandas
df = pandas.read_csv("pima-data-orig.csv")

import sklearn
# subset = df[['glucose_conc','diastolic_bp','insulin','bmi','diab_pred','age','diabetes']]
array = df.values
X = array[:, 0:8]
y = array[:, 8]

# Use SMOTE to make the data even
from imblearn.over_sampling import SMOTE
sm = SMOTE(ratio='auto',kind='regular',random_state=42)
X_sampled, y_sampled = sm.fit_sample(X,y)


# split the data
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_sampled, y_sampled, test_size=0.30, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# train data
model = GaussianNB()
model.fit(X_train, y_train)

print("Done training 70% of data")

# test model
predictions = model.predict(X_test)
print(predictions)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predictions))