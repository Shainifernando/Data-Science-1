import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Reading the data set.
Global_superstore_data = pd.read_excel("Global Superstore Lite.xlsx")

# MBA findings for each segment
for segment in Global_superstore_data["Segment"].unique():
    segment_data = Global_superstore_data[Global_superstore_data["Segment"] == segment]
    
    s = (segment_data.groupby(["Order ID", "Sub-Category"])["Quantity"]
         .sum().unstack().reset_index().fillna(0)
         .set_index("Order ID"))
    
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
