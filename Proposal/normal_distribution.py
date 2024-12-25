import pandas as pd
from scipy.stats import shapiro
import matplotlib.pyplot as plt

data = pd.read_csv('DM_project_24.csv')

normality_results = {}

for column in data.columns:
    column_data = data[column].dropna()

    if column_data.dtype in ['float64', 'int64']:
        stat, p_value = shapiro(column_data)
        normality_results[column] = p_value > 0.05

normality_results_df = pd.DataFrame.from_dict(normality_results, orient='index', columns=['Is Normal'])

normal_count = normality_results_df['Is Normal'].sum()
non_normal_count = len(normality_results_df) - normal_count

plt.figure(figsize=(8, 5))
plt.bar(['Normal Distribution', 'Not Normal Distribution'], [normal_count, non_normal_count], color=['green', 'skyblue'])
plt.ylabel('Number of Columns')
plt.show()