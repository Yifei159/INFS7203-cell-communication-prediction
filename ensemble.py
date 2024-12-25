import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# load data
data = pd.read_csv('data/processed_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# select model, using the best hyperparameters found by each of them by Grid Search
dt = DecisionTreeClassifier(max_depth=11, random_state=42)
rf = RandomForestClassifier(n_estimators=53, max_depth=26, random_state=42)
knn = KNeighborsClassifier(n_neighbors=9)
nb = GaussianNB()

# train model
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
knn.fit(X_train, y_train)
nb.fit(X_train, y_train)

# create voting model, the weight is the accuracy evaluated by each of them by cross validation
voting = VotingClassifier(estimators=[('decision_tree', dt), ('random_forest', rf), ('knn', knn), ('naive_bayes', nb)], voting='soft', weights=[0.9447, 0.9526, 0.9559, 0])
voting.fit(X_train, y_train)

# evaluation
accuracy_scorer = make_scorer(accuracy_score)
f1_scorer = make_scorer(f1_score, average='macro')

# 5 fold cross validation
accuracy_cv = cross_val_score(voting, X, y, cv=5, scoring=accuracy_scorer)
f1_score_cv = cross_val_score(voting, X, y, cv=5, scoring=f1_scorer)

print(f"accuracy: {accuracy_cv.mean():.4f}")
print(f"f1_score: {f1_score_cv.mean():.4f}")

# draw the confusion matrix
y_pre = voting.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pre)
tn, fp, fn, tp = conf_matrix.ravel()
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix for Ensemble Model")
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('pic/ensemble_model.png')
plt.show()