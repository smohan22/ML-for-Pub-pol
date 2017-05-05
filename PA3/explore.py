import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

def read_data(file_name, index_col = None):
    '''Read the data from the file location
        file_name = name of the file to be opened
        index_col = name of the col that we want to make our index
        returns pandas dataframe'''
    data = pd.read_csv(file_name, index_col = index_col)
    
    return data 


def split_data(data, dependent_var, test_size):
    '''
    Split the data set into testing and training data

    data: data to be split_data
    dependent_var: variable that is to be predicted
    test_size: the ratio in which the test and training should be split_data

    returns the split data
    
    '''
        
    X = data.drop([dependent_var], 1)
    y = data[dependent_var]
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size= test_size)
        
    return X_train, X_test, y_train, y_test

def plot_hist(df,df_cols):
    '''
    draw histograms for all the data

    df: our entire data
    df_cols: list of columns for which we need to graph the histograms

    draws the histograms
    '''
    for each in df_cols:
        print("\n")
        #df=data[each].value_counts()
        #meanval = int(data[[each]].mean())
        data[each].plot.hist(align = 'mid')
        print("Graph of ", each)
        plt.xlabel(each)
        #plt.ylabel("Num_people")
        plt.show()
        
def plot_bar(df,df_cols):
    '''
    plot bar chart for all the data

    df: our entire data
    df_cols: list of columns for which we need to graph the histograms

    plot bar charts
    
    '''
    for each in df_cols:
        print("\n")
        df=data[each].value_counts()
        df.plot.bar()
        print("Graph of ", each)
        plt.xlabel(each)
        #plt.ylabel("No_people")
        plt.show()

def plot_corr(df,size=10):
    '''
    Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, vmin=-1, vmax=1, cmap = plt.cm.Greens)
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical');
    plt.yticks(range(len(corr.columns)), corr.columns);
    ticks = np.arange(0,9,1)
    plt.legend(prop={'size':5})
    plt.show()    
        
def basic_exploration(data, dependent_var):
    '''
    describe the data
    find average details depending on dependent_var
    
    dataset: data to be split_data
    dependent_var: variable that is to be predicted
    '''
    
    print(data.describe())
    
    print(data.groupby(dependent_var).mean().transpose())
    
'''
#Doesnt work correctly.
def plot_bar_notworking(df):
    count = 0
    for column in data.columns:
        #max_range = len(data[[columns]])
        #y = [x[0] for x in data[[columns]].values.tolist()]
        w = 0.5
        count = count+1
        plt.bar(np.arange(0,data.shape[0])+ (w*count), list(data[column]), width = w, align='center', color = np.random.rand(3,1), label = column)
        plt.xticks(range(data.shape[0]))
    plt.legend(prop={'size':5})
    #plt.show()
'''