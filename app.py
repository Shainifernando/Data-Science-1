import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

Global_superstore_data = pd.read_excel("Global Superstore Lite.xlsx") 
