
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# warnings
import warnings
warnings.filterwarnings("ignore")


healthcare=pd.read_csv("C:/Users/manis/health care diabetes.csv")

features=healthcare.drop('Outcome',axis=1)
labels=healthcare['Outcome']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features, labels, test_size=0.2, random_state=10)


#GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)
y_pred = gbc.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)

print(classification_report(y_test,y_pred))



pickle.dump(gbc, open('diabetic.pkl', 'wb'))


















