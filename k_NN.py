import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# load data
data = pd.read_csv('data/processed_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create kNN
knn = KNeighborsClassifier()

# train model
# knn.fit(X_train, y_train)

# use Grid Search to find the best hyperparameter n_estimators
f1_scorer = make_scorer(f1_score, average='macro')
hp = {'n_neighbors': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}
grid = GridSearchCV(estimator=knn, param_grid=hp, scoring=f1_scorer, cv=5)
grid.fit(X, y)
best_k = grid.best_estimator_

# evaluation
accuracy_scorer = make_scorer(accuracy_score)
f1_scorer = make_scorer(f1_score, average='macro')

# 5 fold cross validation
accuracy_cv = cross_val_score(best_k, X, y, cv=5, scoring=accuracy_scorer)
f1_score_cv = cross_val_score(best_k, X, y, cv=5, scoring=f1_scorer)

print(f"the best k: {grid.best_params_['n_neighbors']}")
print(f"accuracy: {accuracy_cv.mean():.4f}")
print(f"f1_score: {f1_score_cv.mean():.4f}")

# draw the confusion matrix
y_pre = best_k.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pre)
tn, fp, fn, tp = conf_matrix.ravel()
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix for k-NN")
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('pic/k_NN.png')
plt.show()