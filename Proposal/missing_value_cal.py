import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('DM_project_24.csv')

count = data.isnull().sum()

plt.figure(figsize=(16, 6))
count.plot(kind='bar')
plt.xticks(rotation=90, fontsize=9)
plt.tight_layout()
plt.xlabel("Features")
plt.ylabel("Number of Missing Values")

plt.savefig('/Users/yifeiwei/Desktop/missing_value.png')
plt.show()