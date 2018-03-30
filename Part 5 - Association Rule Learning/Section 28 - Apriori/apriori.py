import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None).as_matrix()

transactions = []
for items_in_transaction in dataset:
    transaction = []
    for item in items_in_transaction:
        item = str(item)
        #np.isnan cannot be used here cause not empty values are strings
        if item == 'nan':
            continue
        transaction.append(item)
    transactions.append(transaction)

# products bought at least 3 times a week
min_support = 3 * 7 / len(dataset)
min_confidence = 0.4
min_lift = 3
rules = apriori(
    transactions,
    min_support=min_support,
    min_confidence=min_confidence,
    min_lift=min_lift,
    min_length=3
)

rules = list(rules)
