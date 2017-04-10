import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

data = pd.read_csv('credit_card_fraud.csv')

print('---- data (head)')
print(data.head())

print('---- categorical features')
print(data.fraudulent.value_counts())
print(data.card_country.value_counts())

print('---- encoded countries')
encoded_countries = pd.get_dummies(data.card_country, prefix='cc')
print(encoded_countries.head())

data = data.join(encoded_countries)
print('---- joined data')
print(data.head())

y = data.fraudulent
X = data[['amount', 'card_use_24h', 'cc_AU', 'cc_GB', 'cc_US']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# dt_model = DecisionTreeClassifier(max_depth=3, min_samples_split=20).fit(X_train, y_train)
dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=5).fit(X_train, y_train)

print('---- model')
print(dt_model.tree_)
# print(lr_model.intercept_)

print('---- test')
y_test_predict_lr = dt_model.predict_proba(X_test)
# print(y_test_predict_lr)
# print(lr_model.classes_)
y_test_scores_lr = [x[1] for x in y_test_predict_lr]

fpr, tpr, thresholds = roc_curve(y_test, y_test_scores_lr)
auc_score = roc_auc_score(y_test, y_test_scores_lr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
