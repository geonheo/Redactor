from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from utils import drop_if, extract_if
import pandas as pd
import numpy as np
import copy
from datetime import datetime
from scipy.stats import pearsonr

def normalize_and_split_dataset(data, label, tsize, target_i):
    """
    Split the dataset into train, test, and target dataset and normalize them with respect to only training dataset.
    
    Args:
        data: all dataset used in the whole experiments
        label: corresponding true labels of the data
        tsize: the relative rate of test dataset compared to whole dataset
        target_i: indices of target points
    """
    
    xta, yta = data.loc[target_i], label.loc[target_i].values
    xtr, xte, ytr, yte = train_test_split(data, label.values, test_size  = tsize)
    scaler = StandardScaler()  
    scaler.fit(xtr)  
    xtr, xte, xta = scaler.transform(xtr), scaler.transform(xte), scaler.transform(xta)
    return (xtr, ytr[:,0]), (xte, yte[:,0]), (xta, yta[:,0]), scaler

class Dataset:
    def __init__(self, dataset):
        """
        Args:
            dataset: dataset name
        """
        if dataset == 'adult':
            self.data_1hot, self.label = self.load_adult_data()
        elif dataset == 'compas':
            self.data_1hot, self.label = self.load_compas_data()
        
    def split_dataset(self, test_size, target_idxs):
        """
        Split the dataset into train, test, and target dataset and normalize them with respect to only training dataset.
    
        Args:
            test_size: the relative rate of test dataset compared to whole dataset
            target_idxs: indices of target points
        """
        
        return normalize_and_split_dataset(self.data_1hot, self.label, test_size, target_idxs)
    
    def load_adult_data(self):
        """
        Load AdultCensus dataset.
        """
        
        self.column_cat = ['workclass', 'education', 'marital-status', 'occupation', 
                           'relationship', 'race', 'gender', 'native-country']
        self.column_int = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        
        column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
                        'marital-status', 'occupation', 'relationship', 'race', 'gender',
                        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country','income']

        train = pd.read_csv('./dataset/adult_data.txt', sep=",\s", header=None, names = column_names, engine = 'python')
        test = pd.read_csv('./dataset/adult_test.txt', sep=",\s", header=None, names = column_names, engine = 'python')
        test['income'].replace(regex=True,inplace=True,to_replace=r'\.',value=r'')

        adult_all = pd.concat([test,train])
        adult_all.reset_index(inplace = True, drop = True)
        
        for c in self.column_cat+['income',]:
            tmp_idx = adult_all[adult_all[c]=='?'].index
            adult_all = adult_all.drop(tmp_idx)
            
        adult_cat_1hot = pd.get_dummies(adult_all.select_dtypes('object'))
        adult_non_cat = adult_all.select_dtypes(exclude = 'object')
        adult_all_1hot = pd.concat([adult_non_cat, adult_cat_1hot], axis=1, join='inner')
        adult_all_1hot = drop_if(adult_all_1hot, '?')

        idxs = ['i%s' % i for i in range(len(adult_all_1hot))]
        adult_all_1hot.index = idxs
        adult_all.index = idxs
        
        self.data = drop_if(adult_all, 'income')
        
        adult_data = drop_if(adult_all_1hot, 'income')
        adult_label = extract_if(adult_all_1hot, 'income')
        
        return adult_data, adult_label[['income_>50K']]
    

    def load_compas_data(self):
        """
        Load COMPAS dataset.
        """
        
        raw_data = pd.read_csv('./dataset/compas-scores-two-years.csv')
        idxs = ['i%s' % i for i in range(len(raw_data))]
        raw_data.index = idxs
        raw_data.columns = [c.replace('_','-') for c in raw_data.columns]
        
        df = raw_data[((raw_data['days-b-screening-arrest'] <=30) & 
              (raw_data['days-b-screening-arrest'] >= -30) &
              (raw_data['is-recid'] != -1) &
              (raw_data['c-charge-degree'] != 'O') & 
              (raw_data['score-text'] != 'N/A')
             )]
        def date_from_str(s):
            return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
        
        df['length-of-stay'] = (df['c-jail-out'].apply(date_from_str)-df['c-jail-in'].apply(date_from_str)).dt.total_seconds()
        self.column_cat = ['c-charge-degree', 'age-cat', 'race', 'sex']
        self.column_int = ['age', 'juv-fel-count', 'juv-misd-count', 'juv-other-count', 
                           'priors-count', 'two-year-recid']
        column_names = self.column_cat+self.column_int
        
        self.data = pd.concat([df[c] for c in column_names],axis=1)
        
        def flatten_compas(DF, col_int):
            df_cat1 = pd.get_dummies(DF[['c-charge-degree', 'sex']])
            df_cat2 = pd.get_dummies(DF[['age-cat', 'race']])
           
            df_int = [DF[c] for c in col_int]
            data = pd.concat([df_cat1, df_cat2]+df_int, axis=1)
            return data
        
        compas_data = flatten_compas(self.data, self.column_int)
        df_score = pd.get_dummies(df['score-text'] != 'Low',drop_first=True)
        compas_label = pd.DataFrame(df_score.values.ravel(), index=compas_data.index)
        
        return compas_data, compas_label


        