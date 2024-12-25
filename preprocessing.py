import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor

# load data
data = pd.read_csv('data/DM_project_24.csv')

# devide data
feature = data.columns[:105]
target = data.columns[105]
numerical_data = data.columns[:103]
nominal_data = data.columns[103:105]

# the numerical missing data can be filled by the mean value in the cluster
imputer_num = KNNImputer(n_neighbors=7)
data[feature] = imputer_num.fit_transform(data[feature])

# the nominal data should be 1 or 0, so based on the filled value change the value>=0.5 to 1, and the value<0.5 to 0
for col in nominal_data:
    data[col] = [1 if x >= 0.5 else 0 for x in data[col]]

# use lof method to wipe outliers
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
outliers = lof.fit_predict(data[feature])
data = data[outliers == 1]

# max-min normalization
data[numerical_data] = (data[numerical_data] - data[numerical_data].min()) / (data[numerical_data].max() - data[numerical_data].min())

data.to_csv('data/processed_data.csv', index=False)

print(f'still {data.isnull().sum().sum()} missing values left')
print(f'wiped {(outliers == -1).sum()} outliers')