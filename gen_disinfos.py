from ctgan import CTGANSynthesizer
from ctgan import load_demo
from utils import flatten_data, shorten_data
from tqdm import tqdm
import copy
import pandas as pd
import numpy as np

def abnormal_examples (df, col1, col2, th=1e-4):
    """
    Find abnormal examples using confusion matrix.
    
    Args:
        df: original DataFrame instance
        col1 & col2: 2 columns of df
    """
    
    abnormal = []
    confusion_matrix = pd.crosstab(df[col1], df[col2])/len(df) <= th
    val1, val2 = confusion_matrix.index, confusion_matrix.columns
    
    for v2 in val2:
        tmp = val1[confusion_matrix[v2]]
        abnormal.extend([{col1:t, col2:v2} for t in tmp])        
    return abnormal

def filter_table(df, d):
    """
    Filter the abnormal examples from original data.
    
    Args:
        df: original DataFrame instance
        d: abnormal feature value pairs
    """
    
    cond1 = lambda x: (x[list(d.keys())[0]] == list(d.values())[0]).astype(int)
    cond2 = lambda x: (x[list(d.keys())[1]] == list(d.values())[1]).astype(int)
    return df[(cond1(df)+cond2(df)) == 2]

class GANcandidates:
    """
    Class for generating candidate points using the GAN model
    """
    
    def __init__(self):
        self.gan = CTGANSynthesizer(epochs=100)
        
    def fit(self, data, col_cat, col_int):
        """
        Train the CTGAN model on all data.
        
        Args:
            data: original data without label
            col_cat: categorical columns
            col_int: integer columns
        """
        
        self.gan.fit(data, col_cat)
        self.data = data
        self.col_cat = col_cat
        self.col_int = col_int
        
    def generate(self, n_gen=int(1e+5), col_cand=None):
        """
        Generate fake candidates using trained the CTGAN model.
        
        Args:
            n_gen: the number of initial datapoints using CTGAN
            col_cand: columns to check abnormality
        """
        
        samples = self.gan.sample(n_gen)
        self.generated = self.preprocessing(samples, col_cand)
        self.generated.index = ['g%s' % str(i) for i in self.generated.index]
        return self.generated
    
    def preprocessing(self, samples, col_cand):
        """
        Remove unrealistic points from all generated data using confusion matrix.
        
        Args:
            samples: generated samples using GAN
            col_cand: columns to check abnormality
        """
        
        if col_cand == None:
            col_cand = self.col_cat
            
        filter_list = []
        for i, c1 in enumerate(col_cand):
            for c2 in col_cand[i+1:]:
                filter_list.extend(abnormal_examples(self.data, c1, c2))
                
        samples_clean = copy.deepcopy(samples)
        for f in filter_list:
            abnormals = filter_table(samples_clean, f)
            samples_clean = samples_clean.drop(abnormals.index)
        
        for col in self.col_int:
            max_val, min_val = self.data[col].max(), self.data[col].min()
            samples_clean[col].loc[samples_clean[col] < min_val] = min_val
            samples_clean[col].loc[samples_clean[col] > max_val] = max_val
            
        return samples_clean
    
    def nearest_points(self, scaler, target_indices, col_list, dnum=None):
        """
        Select nearest points around the target points
        
        Args:
            scaler: standard scaler for normalization
            target_indices: indices of the target points
            col_list: columns of flatten data (one-hot encoding)
        """
        
        if dnum == None:
            dnum = len(self.generated)//1000

        tdata = flatten_data(self.data.loc[target_indices], col_list)
        tdata = scaler.transform(tdata)

        nearest = []
        gen_1hot = flatten_data(self.generated, col_list)
        gen_normed = scaler.transform(gen_1hot)

        for t in tdata:
            dist = np.ones(len(gen_normed))*np.inf
            for gi in range(len(gen_normed)):
                dist[gi] = np.linalg.norm(gen_normed[gi]-t)
            nearest.append(self.generated.iloc[np.argsort(dist)[:dnum]])

        return nearest

    
class WMcandidates:
    """
    Class for generating candidate points using the Watermarking technique.
    """
    
    def __init__(self, data_1hot, label, target_indices):
        """
        Args:
            data_1hot: one-hot encoded original data (DataFrame)
            label: labels of original data
            target_indices: indices of the target points
        """
        
        self.data_1hot = data_1hot
        self.label = label
        self.target_indices = target_indices
        
    def watermarking(self, scaler, col_list, col_cat, col_int, clipping=True):
        """
        Apply watermarking technique between the target point and base points.
        
        Args:
            scaler: standard scaler for normalization
            col_list: columns of flatten data (one-hot encoding)
            col_cat: categorical columns
            col_int: integer columns
        """
        
        linear_cands = []
        r = lambda x : round(x+0.001)
        
        for ti in tqdm(self.target_indices):
            tdata, tlabel = self.data_1hot.loc[ti], self.label.loc[ti]
            target_norm = scaler.transform([tdata])
            cand_idx, dist = [], []
            for ii in self.data_1hot.index:
                cand, cand_label = self.data_1hot.loc[ii], self.label.loc[ii]
                if np.sum(cand_label==tlabel) == 0:
                    cand_norm = scaler.transform([cand])
                    dist.append(np.linalg.norm(target_norm-cand_norm))
                    cand_idx.append(ii)
                    
            sorted_idx = np.argsort(dist)
            cand_idx = np.array(cand_idx)
            base = self.data_1hot.loc[cand_idx[sorted_idx][1]]
            base_norm = scaler.transform([base])
            n_dist = np.linalg.norm(target_norm-base_norm)

            lin = []
            for li in range(1,30):
                gamma = li/30
                linear_num = (1-gamma)*tdata+(gamma)*base
                linear = r(linear_num) if clipping else linear_num
                linear_norm = scaler.transform([linear])
                lin.append(linear)
                
            idxs = []

            lin = pd.DataFrame(np.array(lin).astype(int), columns=self.data_1hot.columns)
            lin.index = ['w%s' % str(i) for i in lin.index]
            linear_cands.append(shorten_data(lin, col_list, col_cat, col_int))  
        return linear_cands
        
        
def agg_disinfo(pdb, cands, scaler, xtr, ytr, tdata, tlabel, col_list, n_disinfo=500, dup=True, alpha=0.7):
    """
    Select the best disinformation instance among all candidate points and
    normalize them to put into the dataset.
    
    Args:
        pdb: PDB instance which uses surrogate models and the generative model
        cands: all candidate points 
        scaler: standard scaler for normalization
        tdata: target datapoint
        tlabel: the label of the target data
        col_list: columns of flatten data (one-hot encoding)
        n_disinfo: the number of disinformation
        dup: If True, duplicates of the best disinformation are generated
        alpha: confidence threshold for probabilistic decision boundary
    """
    
    cands_1hot = flatten_data(cands, col_list)
    cands_normed = scaler.transform(cands_1hot)
    probs = pdb.predict_proba(cands_normed)
    
    conf_idx = probs[:,tlabel]<(1-alpha)
    
    if sum(conf_idx) == 0:
        dist = np.ones(len(xtr))*np.inf
        for i in range(len(xtr)):
            if ytr[i] != tlabel:
                dist[i] = np.linalg.norm(xtr[i]-tdata)
                
        if dup:
            nearest = xtr[np.argsort(dist)[0]]
            disinfos = np.concatenate([[nearest]]*n_disinfo, axis=0)
            y = ytr[np.argsort(dist)[0]]
            disinfo_label = np.concatenate([[y]]*n_disinfo, axis=0)
        else:
            disinfos = xtr[np.argsort(dist)[:n_disinfo]]
            disinfo_label = ytr[np.argsort(dist)[:n_disinfo]]
         
    else:
        dist = np.ones(len(cands))*np.inf
        for ci in range(len(cands)):
            if conf_idx[ci]:
                dist[ci] = np.linalg.norm(cands_normed[ci]-tdata)
            
        if dup:
            nearest = cands_normed[np.argsort(dist)[0]]
            disinfos = np.concatenate([[nearest]]*n_disinfo, axis=0)
            y = pdb.predict([nearest])
            disinfo_label = np.concatenate([[y]]*n_disinfo, axis=0)
        else:
            disinfos = cands_normed[np.argsort(dist)[:n_disinfo]]
            disinfo_label = pdb.predict(disinfos)
        
    return disinfos, disinfo_label.astype(int)
        
        
