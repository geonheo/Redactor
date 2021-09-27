import pandas as pd
import numpy as np

def drop_if(df, char):
    """
    Drop the columns that contain 'char' from 'df'
    
    Args:
        df: DataFrame instance
        char: characters in the columns to drop
    """
    
    remove = []
    for c in df.columns:
        if char in c.split('_'):
            remove.append(c)
    return df.drop(columns = remove)

def extract_if(df, char):
    """
    Extract the columns that contain 'char' from 'df'
    
    Args:
        df: DataFrame instance
        char: characters in the columns to extract
    """
    
    remove = []
    for c in df.columns:
        if char in c.split('_'):
            remove.append(c)
    return df[remove]

def flatten_data(df, col_list):
    """
    Convert categorical columns to one-hot integer columns.
    
    Args:
        df: DataFrame instance
        col_list: columns of flatten data (one-hot encoding)
    """
    
    data_cat = pd.get_dummies(df.select_dtypes('object'))
    data_int = df.select_dtypes(exclude = 'object')
    data_1hot = pd.concat([data_cat, data_int], axis=1, join='inner')
    data_1hot.index = np.arange(len(data_1hot))
    d = pd.DataFrame(0, index=np.arange(len(data_1hot)), columns=col_list)
    for c in data_1hot.columns:
        if c in col_list:
            d[c] = data_1hot[c]
    return d

def shorten_data(df, col_list, col_cat, col_int):
    """
    Convert one-hot columns to categorical columns.
    
    Args:
        df: DataFrame instance
        col_list: columns of flatten data (one-hot encoding)
        col_cat: categorical columns
        col_int: integer columns
    """
    
    new = pd.DataFrame(0, columns=col_list, index=df.index)
    for c in col_int:
        new[c] = df[c]
    
    for c in col_cat:
        x = extract_if(df, c)
        new[c] = x.columns[np.argmax(x)].split('_')[-1]
    return new

def data_num2cat(data, cols):
    """
    Convert integer columns to categorical columns.
    
    Args:
        data: DataFrame instance
        cols: integer columns to be converted
    """
    
    tmp = copy.deepcopy(data)
    class_num = 10

    for c in cols:
        st= np.linspace(data[c].min(), data[c].max(),class_num)
        st[0] -= 0.01
        for i in range(class_num-1):
            cond = (st[i]<data[c]) & (data[c]<=st[i+1])
            tmp[c].loc[cond] = str(i)
    return tmp