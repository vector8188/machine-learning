
import numpy as np
from IPython import embed
from housing import load_housing_data
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import matplotlib.pyplot as plt



def split_train_test(data, test_ratio):
    shuffle_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffle_indices[:test_set_size]
    train_indices = shuffle_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


housing = load_housing_data()
housing['income_cat'] = np.ceil(housing['median_income']/1.5)
housing['income_cat'].where(housing['income_cat']<5, 5.0,  inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]



strat_test_set["income_cat"].value_counts()/len(strat_test_set)
housing = strat_train_set.copy()
corr_matirx = housing.corr()
corr_matirx["median_house_value"].sort_values(ascending=False)
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
#%%
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.2)
plt.show()

