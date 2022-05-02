# Importing the packages for data analyses purpose

import pandas as pd
import wordcloud
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# dataframe reading through a processed csv file obtained from preprocess.py

data_v2 = pd.read_csv('dataset-final-processed.csv', names=["id", "sentiment", "review"])

# checking dataset information

print(data_v2.info())

# checking dataset detailed statistics

print(data_v2.describe())

# checking dataset columns

print(data_v2.columns)

# checking dataset unique entries

print(data_v2['sentiment'].unique())

# checking dataset unique entries count for positives and negatives

print(data_v2['sentiment'].value_counts())

#plotting the figure

plt.figure(figsize=(8,5))
sns.countplot(x = data_v2.sentiment)
plt.show()

