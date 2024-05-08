import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#reading the dataset.
Global_superstore_data = pd.read_excel("Global Superstore Lite.xlsx")

# relationship analysis
corelation = Global_superstore_data.corr()
sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns, annot=True)

# Reading the cleaned dataset
df = pd.read_excel('cleaned_dataset_global(1).xlsx')

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
print(rules_s1)

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
print(rules_s2)

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
print(rules_s3)

# vis3(heatmap)
heatmap_data = rules_s3.pivot(index='antecedents', columns='consequents', values='lift')
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('lift Heatmap of Association Rules')
plt.xlabel('Consequents')
plt.ylabel('Antecedents')
plt.show()

# MBA findings for each segment
for segment in Global_superstore_data["Segment"].unique():
    segment_data = Global_superstore_data[Global_superstore_data["Segment"] == segment]
    
    s = (segment_data.groupby(["Order ID", "Sub-Category"])["Quantity"]
         .sum().unstack().reset_index().fillna(0)
         .set_index("Order ID"))
    
    # Convert Order ID to string to avoid conversion to float error
    s.index = s.index.astype(str)
    
    s_sets = s.applymap(lambda x: 0 if x <= 0 else 1)
    
    frequent_itemsets = apriori(s_sets, min_support=0.001, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    
    heatmap_data = rules.pivot(index='antecedents', columns='consequents', values='lift')
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'lift Heatmap of Association Rules for Segment: {segment}')
    plt.xlabel('Consequents')
    plt.ylabel('Antecedents')
    plt.show()

    # Unique results
    unique_results = rules[(rules["lift"] >= 1) & (rules["confidence"] >= 0.1)]
    print(unique_results)

    # Scatter plot
    plt.figure(figsize=(5, 4))
    plt.scatter(range(len(unique_results)), unique_results['lift'], c=unique_results['lift'], cmap='coolwarm')
    plt.colorbar(label='Lift')
    plt.title(f'Scatter Plot of Lift Values for Segment: {segment}')
    plt.xlabel('Association Rule Index')
    plt.ylabel('Lift')
    plt.show()
