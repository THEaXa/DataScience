import pandas
df = pandas.read_csv('census_income_dataset.csv')

import sklearn
subset = df[['age','workclass','education','marital_status','occupation','race','sex','hours_per_week','income_level']]

subset['workclass'].replace({'Private':0,'Self-emp-not-inc':1,'Local-gov':2,'State-gov' : 3,'Self-emp-inc' :4,'Federal-gov':5,'Without-pay':6,'Never-worked':7,'?' : 8}, inplace=True)
subset['education'].replace({'10th':0,'11th':1,'12th':2,'1st-4th' : 3,'5th-6th' : 4,'7th-8th' :5,'9th':6,'Assoc-acdm':7,'Assoc-voc':8,'Bachelors':9,'Doctorate':10,'HS-grad':11,'Masters':12,'Assoc-voc':13,'Preschool':14,'Prof-school':15,'Some-college':16}, inplace=True)
subset['marital_status'].replace({'Married-civ-spouse':0,'Never-married':1,'Divorced':2,'Separated' : 3,'Widowed' :4,'Married-spouse-absent':5,'Married-AF-spouse':6}, inplace=True)
subset['occupation'].replace({'Craft-repair':0,'Armed-Forces':1,'Adm-clerical':2,'Transport-moving' : 3,'Tech-support' :4,'Sales':5,'Protective-serv':6,'Prof-specialty' :7,'Priv-house-serv':8,'Other-service':9,'Machine-op-inspct' :10,'Handlers-cleaners':11,'Farming-fishing':12,'Exec-managerial' :13,'?':14}, inplace=True)
subset['race'].replace({'White':0,'Black':1,'Asian-Pac-Islander':2,'Amer-Indian-Eskimo' : 3,'Other' :4}, inplace=True)
subset['sex'].replace({'Male':0,'Female':1}, inplace=True)

print(subset)

array = subset.values
X = array [:,0:8]
y = array[:,8]

from imblearn.over_sampling import SMOTE
sm = SMOTE(ratio='auto',kind='regular',random_state=42)
X_sampled, y_sampled = sm.fit_sample(X,y)

from sklearn import model_selection
X_train,X_test,y_train,y_test = model_selection.train_test_split(X_sampled,y_sampled, test_size=0.30, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
object =  DecisionTreeClassifier()

print("Model is training...")

object.fit(X_train, y_train)
print("model done training")

predictions = object.predict(X_test)
print(predictions)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predictions))

newperson = [[23,4,9,0,4,1,1,40]]
observe = object.predict(newperson)
print('Predicted: ', observe)

# info@modcom.co.ke

