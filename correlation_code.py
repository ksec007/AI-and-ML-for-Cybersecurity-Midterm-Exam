
X = [2, -1, -4, 1, 1, -2, -3, -2]
Y = [-5, -3, -4, -1, 3, 1, 5, 7]

import math

def pearson_r(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator = math.sqrt(sum((xi - mean_x) ** 2 for xi in x)) * math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    return numerator / denominator

r = pearson_r(X, Y)
print("Pearson correlation coefficient (r):", r)
