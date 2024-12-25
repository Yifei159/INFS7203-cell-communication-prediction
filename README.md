# INFS7203-cell-communication-prediction
This is a full-mark solution of the INFS7203 AKA Data Mining course project in UQ.

## Requirements

The result of this program are based on:

    Operating System: macOS Sequoia (Version 15.0)

    Programming Language: Python 3.9.19 (Coded using Pycharm)

To run this software requires Python 3.8 or higher. To install the dependencies, run::

    pip install -r requirements.txt

The main functionality of this software also requires the DM_project_24.csv
datasets.

## Usage

In this section, the various usages are described. Using this software, the
user is able to preprocess the data, including data cleaning and the missing data filling,
train the models, generate predictions using the trained model, and evaluate the predictions.

### Preprocessing


To preprocessing the data, run::

    DM_project/preprocessing.py

The output of this program is the number of missing values that still in the dataset
(which should be 0), and the number of outliers that has been wiped from the dataset.

### Training & Evaluation


To train and evaluate the decision tree, run::

    DM_project/Decision_Tree.py

The output of this program is the best max depth of the trees in this case, which are all found by cross validation
accuracy score and f1-score of decision tree model on this dataset by 5 fold cross validation.

To train and evaluate the random forest, run::

    DM_project/Random_forest.py

The output of this program is the best num of trees in the ramdom forest model and the best max depth of the trees in this
case, which are all found by cross validation, and the accuracy score and f1-score of random forest model on this dataset
by 5 fold cross validation.

To train and evaluate the k_NN, run::

    DM_project/k_NN.py

The output of this program is the best k in the k_NN model in this case, which is found by cross validation, and the
accuracy score and f1-score of k_NN model on this dataset by 5 fold cross validation.

To train and evaluate the Naïve bayes, run::

    DM_project/Naive_Bayes.py

The output of this program is the accuracy score and f1-score of Naïve Bayes model on this dataset by
5 fold cross validation.

To train and evaluate the ensemble model, run::

    DM_project/ensemble.py

The output of this program is the accuracy score and f1-score of the ensemble model on this dataset by
5 fold cross validation. All the hyperparameters are defined by the best
value found by each model program with cross validation. The weight of each models in the ensemble model
is defined by the F1-score of each model calculated in program with 5 folds cross validation while the Naïve 
bayes model was not included, due to the low performance in this case.

After running the program above, the best hyperparameter for each model can be got. Then the best hyperparameters 
are put in this main program to apply the models on the test dataset, run::

    DM_project/main.py

The output of this program is a txt file, which contains the result of applying the ensemble model on the test
dataset provided in the 2nd phase of the data mining project. The txt file was named s4857961.txt.

## Experiment & Result

The result of the experiment and how it's get are described in this section. Including the hyperparameters and 
the weights of each models in the ensemble model.

### Hyperparameters

The hyperparameters in the main prediction file are defined by the GridSearchCV in each model file, which are as followed.

Decision Tree:

    max_depth=11

Random Forest:

    n_estimators=53, max_depth=26

KNN:

    n_neighbors=9

### Weight

The weights of the three models used in the ensemble model are:

    Decision Tree: 0.7458
    Random Forest: 0.7740
    KNN: 0.7783

The weights were defined by the accuracy score of each model calculated by cross validation.
Naïve Bayes was not included in the final ensemble model, due to the low performance.

### Result

The evaluation results of each model are:

Decision Tree:

    Accuracy: 0.9447
    F1-score: 0.7458

Random Forest:

    Accuracy: 0.9526
    F1-score: 0.7740

KNN:

    Accuracy: 0.9559
    F1-score: 0.7783

Naïve Bayes:

    Accuracy: 0.9059
    F1-score: 0.6496

Ensemble Model:

    Accuracy: 0.9546
    F1-score: 0.7816

All the accuracy and F1 score are measured by cross validation.

## Reference

The following are links to the documentation of packages used in the code:

1.	Pandas: https://pandas.pydata.org/docs/

2.	Train-test-split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

3.	Decision Tree Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

4.	Random Forest Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

5.	Voting Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html

6.	K-nearest Neighbours Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

7.	Naive Bayes Classifier (GaussianNB): https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

8.	Cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html

9.	Accuracy Score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

10.	F1 Score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

11.	Confusion Matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

12.	Confusion Matrix Display: https://scikit-learn.org/stable/whats_new/v0.24.html#id7

13.	GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

14.	KNN Imputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html

15.	Local Outlier Factor: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
