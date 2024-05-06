import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# reading the data set.
Global_superstore_data = pd.read_excel("Global Superstore Lite.xlsx") 

#relationship analysis.
corelation = Global_superstore_data.corr()
sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns, annot=True)

#MBA findings
#segment1
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





