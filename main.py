import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score

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

# 5 fold cross validation
accuracy_scorer = make_scorer(accuracy_score)
f1_scorer = make_scorer(f1_score, average='macro')

accuracy_cv = cross_val_score(voting, X, y, cv=5, scoring=accuracy_scorer)
f1_score_cv = cross_val_score(voting, X, y, cv=5, scoring=f1_scorer)

# apply the model on the test dataset
test_data = pd.read_csv('data/test_data.csv')

y_pred = voting.predict(test_data)

# write the result to the file
with open('s4857961.txt', 'w') as f:
    for prediction in y_pred:
        f.write(f"{prediction},\n")
    f.write(f"{accuracy_cv.mean():.3f},{f1_score_cv.mean():.3f}")