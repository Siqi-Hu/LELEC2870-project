from collections import Counter

def reduce_cardinality(dataset, column_name, threshold, inplace=False):

    column = dataset[column_name]
    threshold_freq = int(threshold * len(column))
    counts = Counter(column)

    sum_freq = 0
    categories_kept = []

    for level, freq in counts.most_common():
        sum_freq += dict(counts)[level]
        categories_kept.append(level)

        if sum_freq >= threshold_freq:
            break

    categories_kept.append('Other')

    new_column = column.apply(lambda x: x if x in categories_kept else 'Other')

    if inplace:
        dataset[column_name] = new_column
    else:
        return new_column

# X1_cat['studio'] = reduce_cardinality(dataset=X1, column_name='studio', threshold=0.72, inplace=False)