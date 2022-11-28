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
        elif dataset == 'ESR':
            self.data_1hot, self.label = self.load_ESR_data()
        elif dataset == 'diabetes':
            self.data_1hot, self.label = self.load_diabetes_data()
        
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
    
    def load_ESR_data(self):
        """
        Load ESR dataset.
        """
        
        df = pd.read_csv("./dataset/Epileptic_Seizure_Recognition.csv")
        idxs = ['i%s' % i for i in range(len(df))]
        df.index = idxs
        self.data = df.iloc[:,1:179]
        self.column_cat = []
        self.column_int = list(self.data.columns)
        
        ESR_data = df.iloc[:,1:179]
        ESR_label = pd.DataFrame(df.iloc[:,179], index=ESR_data.index)-1
        
        return ESR_data, ESR_label
    
    def load_diabetes_data(self):
        """
        Load Diabetes dataset.
        """
        
        df = pd.read_csv("dataset/diabetic_data.csv")
        df = df.drop_duplicates(subset=['patient_nbr'])
        features_drop_list = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty', 'repaglinide', 
                              'nateglinide', 'chlorpropamide', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 
                              'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
                              'metformin-rosiglitazone','metformin-pioglitazone', 'acetohexamide', 'tolbutamide', 'max_glu_serum']
        df.drop(features_drop_list, axis=1,inplace=True)

        df.loc[df['diag_1'].str.contains('V',na=False,case=False), 'diag_1'] = 0
        df.loc[df['diag_1'].str.contains('E',na=False,case=False), 'diag_1'] = 0
        df.loc[df['diag_2'].str.contains('V',na=False,case=False), 'diag_2'] = 0
        df.loc[df['diag_2'].str.contains('E',na=False,case=False), 'diag_2'] = 0
        df.loc[df['diag_3'].str.contains('V',na=False,case=False), 'diag_3'] = 0
        df.loc[df['diag_3'].str.contains('E',na=False,case=False), 'diag_3'] = 0

        df['diag_1'] = df['diag_1'].replace('?', -1)
        df['diag_2'] = df['diag_2'].replace('?', -1)
        df['diag_3'] = df['diag_3'].replace('?', -1)

        df['diag_1'] = df['diag_1'].astype(float)
        df['diag_2'] = df['diag_2'].astype(float)
        df['diag_3'] = df['diag_3'].astype(float)

        df['race'] = df['race'].replace('?', 'Other')
        df['gender'] = df['gender'].replace('Unknown/Invalid', 'Female')

        for i in range(0,10):
            df['age'] = df['age'].replace('['+str(10*i)+'-'+str(10*(i+1))+')', i*10+5)

        drug_list = ['metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin']
        for i in drug_list:
            df[i] = df[i].replace('No', 0)
            df[i] = df[i].replace('Steady', 2)
            df[i] = df[i].replace('Down', 1)
            df[i] = df[i].replace('Up', 3)

        df['change']=df['change'].replace('No', 0)
        df['change']=df['change'].replace('Ch', 1)

        df['diabetesMed']=df['diabetesMed'].replace('Yes', 1)
        df['diabetesMed']=df['diabetesMed'].replace('No', 0)

        df['readmitted']=df['readmitted'].replace('NO', '<30')
        df['readmitted']=df['readmitted'].replace('>30', 1)
        df['readmitted']=df['readmitted'].replace('<30', 0)
        
        
        subtract_ub = lambda x: x.replace('_','-')
        df.columns = list(map(subtract_ub, df.columns))
        
        def flatten_diabetes(df, cols):
            x = copy.deepcopy(df)
            for c in cols:
                x = pd.concat([x,pd.get_dummies(x[c], prefix=c)], axis=1).drop([c],axis=1)
            return x
        
        self.column_cat = ['race', 'gender', 'A1Cresult']
        self.column_int = ['age', 'admission-type-id', 'discharge-disposition-id', 'admission-source-id', 'time-in-hospital',
                           'num-lab-procedures', 'num-procedures', 'num-medications', 'number-outpatient', 'number-emergency', 
                           'number-inpatient', 'diag-1', 'diag-2', 'diag-3', 'number-diagnoses', 'metformin', 'glimepiride', 
                           'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin', 'change', 'diabetesMed']

        idxs = ['i%s' % i for i in range(len(df))]
        df.index = idxs
        self.data = df.drop(['readmitted'], axis=1)
        diabetes_data = flatten_diabetes(self.data, self.column_cat)
        diabetes_label = df[['readmitted']]

        return diabetes_data, diabetes_label
    
    def sampling(self, n_sample, balance=False):
        if balance:
            idxs = []
            classes = np.unique(self.label)
            
            for c in classes:
                cidxs = list(self.data.index[(self.label == c).values[:,0]])
                sampled_cidxs = np.random.choice(cidxs, n_sample//len(classes), replace=False)
                idxs.extend(list(sampled_cidxs))
        else:
            idxs = np.random.choice(self.data.index, n_sample, replace=False)
        idxs = np.unique(idxs)
        
        self.data_1hot = self.data_1hot.loc[idxs]
        self.data = self.data.loc[idxs]
        self.label = self.label.loc[idxs]

    def nearest_sampling(self, n_sample, target_index, distribution=False, ratio_list=None, oversampling=False):
        scaler = StandardScaler()  
        scaler.fit(self.data_1hot)
        data_1hot_normed = pd.DataFrame(scaler.transform(self.data_1hot), 
                                        index=self.data_1hot.index, 
                                        columns=self.data_1hot.columns)
        target = self.data_1hot.loc[target_index]
        
        if distribution:
            idxs = [target_index]
            classes = np.unique(self.label)
            
            if ratio_list is None:
                ratio_list = [(self.label == c).sum()/len(self.label) for c in classes]
            
            for ii, c in enumerate(classes):
                cidxs = list(self.data.index[(self.label == c).values[:,0]])
                ratio = ratio_list[ii]
                    
                tmp_i, tmp_dist = [], []
                for i in cidxs:
                    if i == target_index:
                        continue
                    d = data_1hot_normed.loc[i]
                    dist = np.linalg.norm(target-d)
                    tmp_dist.append(dist)
                    tmp_i.append(i)

                tmp_idxs = np.argsort(tmp_dist)[:int(n_sample*ratio)]
                if oversampling:
                    tmp_idxs = np.repeat(tmp_idxs, int(max(ratio_list)/ratio))
                idxs.extend(np.array(tmp_i)[tmp_idxs])
        else:
            idxs = [target_index]
            tmp_i, tmp_dist = [], []
            for i in data_1hot_normed.index:
                if i == target_index:
                    continue
                d = data_1hot_normed.loc[i]
                dist = np.linalg.norm(target-d)
                tmp_dist.append(dist)
                tmp_i.append(i)

            tmp_idxs = np.argsort(tmp_dist)[:n_sample]
            if oversampling:
                tmp_idxs = np.repeat(tmp_idxs, int(max(ratio_list)/ratio))
            idxs.extend(np.array(tmp_i)[tmp_idxs])
        
        self.data_1hot = self.data_1hot.loc[idxs]
        self.data = self.data.loc[idxs]
        self.label = self.label.loc[idxs]
        
        
        