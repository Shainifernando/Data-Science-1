import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl as pxl
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#reading the dataset.
Global_superstore_data = pd.read_excel("Global Superstore Lite.xlsx")

# Select only numeric columns for correlation analysis
numeric_columns = Global_superstore_data.select_dtypes(include=np.number)

# relationship analysis
correlation = numeric_columns.corr()
sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns, annot=True)

#Reading the cleaned dataset
df1 = pd.read_excel("/workspaces/Data-Science-1/cleaned_dataset_global (1).xlsx")

# Creating a new column as single_transaction using Customer ID and Order Date
df1["single_transaction"] = df1["Customer ID"].astype(str)+'_'+df1['Order Date'].astype(str)

# Creating a table with the new column and Sub-Category
df2 = pd.crosstab(df1['single_transaction'],df1['Sub-Category'])

###### MBA using Segments to train the dataset
## For Segment 1

s1 = (df[df["Segment"] == "Consumer"]
     .groupby(["Order ID", "Sub-Category"])["Quantity"]
     .sum().unstack().reset_index().fillna(0)
     .set_index("Order ID"))
s1

def encode_units(x):
    if x <=0:
        return 0
    if x >=1:
        return 1

s1_sets = s1.applymap(encode_units)

frequent_itemsets_s1 = apriori(s1_sets, min_support=0.001, use_colnames = True)
rules_s1 = association_rules(frequent_itemsets_s1, metric = "lift", min_threshold=1)

#vis1(heatmap)
heatmap_data = rules_s1.pivot(index='antecedents', columns='consequents', values='lift')
plt.figure(figsize=(5, 4))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('lift Heatmap of Association Rules')
plt.xlabel('Consequents')
plt.ylabel('Antecedents')
plt.show()

## For Segment 2
s2 = (df[df["Segment"] == "Home Office"]
     .groupby(["Order ID", "Sub-Category"])["Quantity"]
     .sum().unstack().reset_index().fillna(0)
     .set_index("Order ID"))
s2

def encode_units(x):
    if x <=0:
        return 0
    if x >=1:
        return 1

s2_sets = s2.applymap(encode_units)

frequent_itemsets_s2 = apriori(s2_sets, min_support=0.001, use_colnames = True)
rules_s2 = association_rules(frequent_itemsets_s2, metric = "lift", min_threshold=1)

# vis2(heatmap)

heatmap_data = rules_s2.pivot(index='antecedents', columns='consequents', values='lift')
plt.figure(figsize=(5, 4))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('lift Heatmap of Association Rules')
plt.xlabel('Consequents')
plt.ylabel('Antecedents')
plt.show()

## For Segment 3

s3 = (df[df["Segment"] == "Corporate"]
     .groupby(["Order ID", "Sub-Category"])["Quantity"]
     .sum().unstack().reset_index().fillna(0)
     .set_index("Order ID"))
s3

def encode_units(x):
    if x <=0:
        return 0
    if x >=1:
        return 1

s3_sets = s3.applymap(encode_units)

frequent_itemsets_s3 = apriori(s3_sets, min_support=0.001, use_colnames = True)
rules_s3 = association_rules(frequent_itemsets_s3, metric = "lift", min_threshold=1)

# vis3(heatmap)
heatmap_data = rules_s3.pivot(index='antecedents', columns='consequents', values='lift')
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('lift Heatmap of Association Rules')
plt.xlabel('Consequents')
plt.ylabel('Antecedents')
plt.show()

## MBA for whole dataset

# Encoding data 
def encode(item_freq):
    res = 0
    if item_freq > 0:
        res = 1
    return res
    
basket_input = df2.applymap(encode)

# Apply apriori algorithm for frequent itemsets
frequent_itemsets = apriori(basket_input, min_support = 0.001, use_colnames=True)

# Generate association rules with lift measure
rules = association_rules(frequent_itemsets, metric = "lift", min_threshold=1)

rules.sort_values(["support", "confidence", "lift"],axis =0, ascending = False)

# Final result vizualization 
# Pivot the DataFrame to prepare it for the heatmap
heatmap_data = rules2.pivot(index='antecedents', columns='consequents', values='lift')

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('lift Heatmap of Association Rules')
plt.xlabel('Consequents')
plt.ylabel('Antecedents')
plt.show()

