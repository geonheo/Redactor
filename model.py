from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy



class Models:
    def define_models(self, mnames, aux_name=None):
        """
        Define models to train and inference.
        
        Args:
            mnames: the name list of the model to use
            aux_name: prefix name for distinguishing models according to the purpose
        """
        
        self.model_names = np.array(['nn_tanh_5_2','nn_tanh_10_2','nn_relu_5_2', 'nn_relu_50_25', 'nn_relu_200_100', 
                                    'nn_relu_25_10','nn_log_5_2', 'nn_identity', 'tree_gini', 
                                    'tree_entropy', 'svm_rbf', 'svm_linear', 'svm_poly', 
                                    'svm_sigmoid', 'rf_gini', 'rf_entropy', 
                                    'gb', 'ada', 'log_reg']) if not mnames else np.array(mnames)
        
        
        
        # Neural Network Models
        nn_tanh_5_2 = MLPClassifier(activation = 'tanh', solver='lbfgs', alpha=1e-1, 
                                 hidden_layer_sizes=(5, 2), random_state=1, warm_start=True)
        nn_tanh_10_2 = MLPClassifier(activation = 'tanh', solver='lbfgs', alpha=1e-1, 
                                 hidden_layer_sizes=(10, 2), random_state=1, warm_start=True)
        nn_relu_5_2 = MLPClassifier(activation = 'relu', solver='adam', alpha=1e-1, 
                                    hidden_layer_sizes=(5, 2), random_state=1, 
                                    learning_rate  = 'invscaling', warm_start = True)
        nn_relu_50_25 = MLPClassifier(activation = 'relu', solver='adam', alpha=1e-3, 
                                      hidden_layer_sizes=(50, 25), random_state=1, 
                                      learning_rate  = 'invscaling', warm_start = True)
        nn_relu_200_100 = MLPClassifier(activation = 'relu', solver='adam', alpha=1e-3, 
                                        hidden_layer_sizes=(200, 100), random_state=1, 
                                        learning_rate  = 'invscaling',warm_start = True)
        nn_relu_25_10 = MLPClassifier(activation = 'relu', solver='adam', alpha=1e-3, 
                                      hidden_layer_sizes=(25, 10), random_state=1, 
                                      learning_rate  = 'invscaling', warm_start = True)
        nn_log_5_2 = MLPClassifier(activation = 'logistic', solver='adam', 
                            alpha=1e-4, hidden_layer_sizes=(5, 2),
                            learning_rate  = 'invscaling', 
                            random_state=1, warm_start = True)
        nn_identity = MLPClassifier(activation = 'identity', solver='adam', alpha=1e-1, 
                                    hidden_layer_sizes=(5, 2), random_state=1, warm_start = True)
        
        # Decision Tree Models
        tree_gini = DecisionTreeClassifier(criterion = 'gini', min_samples_split = 0.05,
                                           min_samples_leaf = 0.001, max_features = None)
        tree_entropy = DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 0.05, 
                                              min_samples_leaf = 0.001)
        
        # SVM Models
        svm_rbf = SVC(kernel = 'rbf', C = 1, tol = 1e-3)
        svm_linear = SVC(kernel = 'linear')
        svm_poly = SVC(kernel = 'poly')
        svm_sigmoid = SVC(kernel = 'sigmoid')
        
        
        # Random Forest Models
        rf_gini = RandomForestClassifier(n_estimators=100, criterion = 'gini', max_features = None,  
                                         min_samples_split = 0.05, min_samples_leaf = 0.001)
        rf_entropy = RandomForestClassifier(n_estimators=100, criterion = 'entropy', max_features = None,  
                                            min_samples_split = 0.05, min_samples_leaf = 0.001)
        
        # Other types of Models
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        ada = AdaBoostClassifier(n_estimators=100)                     
        log_reg = LogisticRegression(penalty = 'l2', dual = False, tol = 1e-4, 
                                     fit_intercept = True, solver = 'liblinear')
        
        self.model_dict = {
            'nn_tanh_5_2' : nn_tanh_5_2,'nn_tanh_10_2' : nn_tanh_10_2, 'nn_relu_5_2' : nn_relu_5_2, 
            'nn_relu_50_25' : nn_relu_50_25, 'nn_relu_200_100' : nn_relu_200_100, 'nn_relu_25_10' : nn_relu_25_10,
            'nn_log_5_2' : nn_log_5_2, 'nn_identity' : nn_identity, 
            'svm_rbf' : svm_rbf, 'svm_linear' : svm_linear, 'svm_poly' : svm_poly, 
            'svm_sigmoid' : svm_sigmoid, 'tree_gini' : tree_gini, 'tree_entropy' : tree_entropy, 'rf_gini' : rf_gini, 
            'rf_entropy' : rf_entropy, 'gb' : gb, 'ada' : ada, 'log_reg' : log_reg
        }
        
        self.models = [self.model_dict[mn] for mn in self.model_names]
        
        if aux_name is not None:
            self.model_names = ['%s-%s' % (aux_name, mn) for mn in self.model_names]
        
        
    def cross_validation(self, x, y, k=5):
        """
        k-cross validation technique to estimate the model performance only using training dataset.
        
        Args:
            x: training data
            y: labels of the training data
            k: parameter for cross validation
        """
        
        result = np.zeros(len(self.model_names))
        kf = KFold(n_splits=k, random_state=None, shuffle=False)

        for tr_idx, val_idx in tqdm(kf.split(x)):
            x_tr, y_tr = x[tr_idx], y[tr_idx]
            x_val, y_val = x[val_idx], y[val_idx]

            cv_models = copy.deepcopy(self.models)
            val_acc = []

            for i, (model, name) in enumerate(zip(cv_models, self.model_names)):
                model.fit(x_tr, y_tr)
                pred = model.predict(x_val)
                result[i] += accuracy_score(y_val, pred)/k
        return result
    
    def show_performance(self, xy_pairs, cnames=None, ptype='acc'):
        """
        Show the performance of the models.
        
        Args:
            xy_pairs: (x, y) pairs to test the model
            cnames: column display names of the performance table
        """
        
        result = np.zeros((len(self.models), len(xy_pairs)))
        for i, (model, name) in tqdm(enumerate(zip(self.models, self.model_names))):
            for j, (x, y) in enumerate(xy_pairs):
                if ptype == 'acc':
                    pred = model.predict(x)
                    result[i,j] = accuracy_score(y, pred)
                elif ptype == 'max_conf':
                    try:
                        conf = model.predict_proba(x).max(axis=1)
                        result[i,j] = np.mean(conf)
                    except:
                        result[i,j] = 0
                elif ptype == 'y_conf':
                    try:
                        conf = model.predict_proba(x)[np.arange(len(y)), y]
                        result[i,j] = np.mean(conf)
                    except:
                        result[i,j] = 0
                elif ptype == 'f1':
                    pred = model.predict(x)
                    result[i,j] = f1_score(y, pred, average='micro')
                elif ptype == 'bin_auc':
                    try:
                        prob = model.predict_proba(x)[:,1]
                        result[i,j] = roc_auc_score(y, prob)
                    except:
                        result[i,j] = 0
                else:
                    print('Invalid "ptype" input')
                    return None
                    
        if cnames == None:
            cnames = ['data%s' % str(i+1) for i in range(len(xy_pairs))]
        return round(pd.DataFrame(result, index=self.model_names, columns=['%s %s' % (c, ptype) for c in cnames]), 4)
    
    def train_all(self, x, y):
        """
        Train all models with x and y.
        
        Args:
            x: training data
            y: labels of the training data
        """
        np.random.seed(123)
        print('train models..')
        for m in tqdm(self.models):
            m.fit(x, y)
            
            
class SurrogateModels(Models):
    def __init__(self, mnames=False):
        """
        Define the models as surrogate models
        
        Args:
            mnames: a list of the model names used as surrogate models
        """
        
        super().define_models(mnames, 's')   

class VictimModels(Models):
    def __init__(self, mnames=False):
        """
        Define the models as victim models
        
        Args:
            mnames: a list of the model names used as victim models
        """
        
        super().define_models(mnames, 'v')   

class AttackModels(Models):
    def __init__(self, mnames=False):
        """
        Define the models as MIA (Membership Inference Attack) models.
        However, in this class, attack models have different input shape from the other models.
        Input size is 3 (output of the victim model+loss) and output of the attack model is the membership.
        
        Args:
            mnames: a list of the model names used as attack models
        """
        
        super().define_models(mnames, 'a') 
        
    
def attack_input(vmodel, xtr, ytr, xte, yte, xta, yta):
    """
    Generate attack inputs for MI attack using all datasets.
    Input size is 3 (output of the victim model+loss) and output of the attack model is the membership.
    
    Args:
        vmodel: victime model
        xtr, ytr: training dataset
        xte, yte: test dataset
        xta, yta: target dataset
    """
    np.random.seed(123)
    
    def cross_entropy(y1, y2):
        def cce(y_true, y_pred):
            one_hot = np.zeros(ytr.max()+1)
            one_hot[y_true] = 1
            return log_loss([one_hot], [y_pred])
        return np.array([cce(y1[i], y2[i]) for i in range(len(y1))])

    labels_train = np.ones(len(ytr))
    labels_test = np.zeros(len(yte))
    labels_target = np.zeros(len(yta))

    for ti in range(len(yta)):
        tr_num = len(np.where((xtr == xta[ti]).all(axis=1))[0])
        te_num = len(np.where((xte == xta[ti]).all(axis=1))[0])
        if tr_num >= te_num:
            labels_target[ti] = 1

    sample_idx = np.random.choice(np.arange(len(xtr)), len(xte))

    prob_train = vmodel.predict_proba(xtr[sample_idx])
    prob_test = vmodel.predict_proba(xte)
    prob_target = vmodel.predict_proba(xta)

    loss_train = cross_entropy(ytr[sample_idx], prob_train)
    loss_test = cross_entropy(yte, prob_test)
    loss_target = cross_entropy(yta, prob_target)

    feature_train = np.c_[prob_train, loss_train]
    feature_test = np.c_[prob_test, loss_test]
    feature_target = np.c_[prob_target, loss_target]
    
    x_mia = np.r_[feature_train, feature_test]
    y_mia = np.r_[labels_train[sample_idx], labels_test]

    return x_mia, y_mia, feature_target, labels_target

        
        
    


