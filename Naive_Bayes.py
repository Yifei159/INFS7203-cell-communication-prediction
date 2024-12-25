import pandas as pd
from sklearn.model_selection import train_test_split
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

# create Naive Bayes
nb = GaussianNB()

# train model
nb.fit(X_train, y_train)

# evaluation
accuracy = make_scorer(accuracy_score)
f1_score = make_scorer(f1_score, average='macro')

# cross validation
accuracy_cv = cross_val_score(nb, X, y, cv=5, scoring=accuracy)
f1_score_cv = cross_val_score(nb, X, y, cv=5, scoring=f1_score)

print(f"accuracy: {accuracy_cv.mean():.4f}")
print(f"f1_score: {f1_score_cv.mean():.4f}")

# draw the confusion matrix
y_pre = nb.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pre)
tn, fp, fn, tp = conf_matrix.ravel()
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix for Naive Bayes")
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('pic/naive_bayes.png')
plt.show()