import pandas as pd

data = [1, 2, 3, 4, 5, "a", "b", "c", "d", "e", True, False]
series = pd.Series(data, index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"])
print(series)