import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

data = pd.read_csv('credit_card_fraud.csv')

print('---- data (head)')
print(data.head())

print('---- fraudulent')
print(data.fraudulent.value_counts())

print('---- card_country')
print(data.card_country.value_counts())

print('---- encoded countries')
encoded_countries = pd.get_dummies(data.card_country, prefix='cc')
print(encoded_countries.head())

data = data.join(encoded_countries)
print('---- joined data')
print(data.head())

y = data.fraudulent
X = data[['amount', 'card_use_24h', 'cc_AU', 'cc_GB']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
lr_model = LogisticRegression().fit(X_train, y_train)

print('---- model')
print(lr_model.coef_)
print(lr_model.intercept_)

print('---- test')
y_test_predict_lr = lr_model.predict_proba(X_test)
# print(y_test_predict_lr)
# print(lr_model.classes_)
y_test_scores_lr = [x[1] for x in y_test_predict_lr]

fpr, tpr, thresholds = roc_curve(y_test, y_test_scores_lr)
auc_score = roc_auc_score(y_test, y_test_scores_lr)
print("AUC score (closer to 1 the better) = " + str(auc_score))