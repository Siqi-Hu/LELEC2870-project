import pandas as pd
import torch
from collections import Counter

def catToFrequency(dataset, column_name, inplace=False):
    column = dataset[column_name]
    counts = Counter(column)

    dict_cat_freq = {}

    for category, freq in counts.most_common():
        dict_cat_freq[category] = freq / len(column)

    return dict_cat_freq

X1 = pd.read_csv("X1.csv")
dict_cat_freq = catToFrequency(X1, 'studio')
torch.save(dict_cat_freq, "studio_freq")

