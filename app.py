import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
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
print('---- prepare data')
print(data.head())

y = data.fraudulent
X = data[['amount', 'card_use_24h', 'cc_AU', 'cc_GB', 'cc_US']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print('---- normalize data')
poly = PolynomialFeatures(2)
X_train_poly = poly.fit_transform(X_train)

scaler = StandardScaler().fit(X_train_poly)
X_train_scaled = scaler.transform(X_train_poly)

print('---- model')
models = []

lr_model = LogisticRegression().fit(X_train_scaled, y_train)
print('logistic regression: coef: ' + str(lr_model.coef_))
print('logistic regression: intercept: ' + str(lr_model.intercept_))
models.append(('Logistic regression', lr_model, True))

dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=5).fit(X_train_scaled, y_train)
models.append(('Random forest', dt_model, True))

clf_model = SGDClassifier(loss="hinge", penalty="l2").fit(X_train_scaled, y_train)
models.append(('SDG', clf_model, False))

print('---- test')
figure = 0
X_test_poly = poly.fit_transform(X_test)
X_test_scaled = scaler.transform(X_test_poly)

for model in models:
    print('Score for %s: %s' % (model[0], model[1].score(X_test_scaled, y_test)))

    if model[2]:
        figure = figure + 1
        y_test_predict_lr = model[1].predict_proba(X_test_scaled)
        y_test_scores_lr = [x[1] for x in y_test_predict_lr]

        fpr, tpr, thresholds = roc_curve(y_test, y_test_scores_lr)
        auc_score = roc_auc_score(y_test, y_test_scores_lr)

        plt.figure(figure)
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (%s)' % model[0])
        plt.legend(loc="lower right")
plt.show()
