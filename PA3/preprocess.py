import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
pl.rcParams['figure.figsize'] = (20,6)

#from datagotham
def print_null_freq(df):
    """
    for a given DataFrame, calculates how many values for 
    each variable is null and prints the resulting table to stdout
    """
    df_lng = pd.melt(df)
    null_variables = df_lng.value.isnull()
    return pd.crosstab(df_lng.variable, null_variables)


def fill_missing(data_set, column_name, strategy = 'mean'):
    '''
    fill the missing data for the required column based on the strategy

    data_set: data_set
    column_name: name of column for which missing data is to be filled
    strategy: the strategy on which missing data is to be filled

    returns the updated DataFrame
    '''
    if strategy == 0:
        data_set[column_name] = data_set[column_name].fillna(0)
    if strategy == "median":
        data_set[column_name] = data_set[column_name].fillna(data_set.median())
    else:
        data_set[column_name] = data_set[column_name].fillna(data_set.mean())
    return data_set


def get_discrete_var(df, col_name, cuts, drop_first = True):
    '''
    Discretizes  continuous variable.
    
    df: pandas dataframe with the continous variable to be discretized
    cuts: number of percentiles
    col_name: (str) to append to dummy columns
    drop_first: (bool) to drop continous variable
    Out:
        - df
    '''
    min_val = df.min()
    max_val = df.max()
    arr = np.arange(0, max_val + cuts, cuts)
    factor = pd.cut(df, arr)
    return pd.get_dummies(factor, prefix = col_name, drop_first = drop_first)

def get_dummy_var(df, col_name, drop_first = False):
    '''
    Takes categorical variable and creates binary/dummy variables from it.
    
    df: pandas dataframe
    col_name: list of categorical variables
    drop_first: (bool) whether or not to drop first dummy
    
    df: pandas dataframe
    '''
    return pd.get_dummies(df, prefix = col_name, drop_first = drop_first)


"""from sklearn.metrics import accuracy_score

#Help taken from https://www.youtube.com/watch?v=yLsKZTWyEDg
def base_rate_model(X):
    y = np.zeros(X.shape[0])
    return y

# How accurate our model utilizing all features is?
y_base_rate = base_rate_model(X_test)
print("Base case accuracy = ",accuracy_score(y_test, y_base_rate))

# Model Accuracy with all features
y_pred_all = model_all.predict(X_test)
print("Model accuracy with all features = ", accuracy_score(y_test, y_pred_all))

# Model Accuracy with top10 features
y_pred_top10 = model_top10.predict(X_test[features])
print("Model accuracy with top 10 features = ", accuracy_score(y_test, y_pred_top10))"""
      