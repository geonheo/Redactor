from tqdm import tqdm
from snorkel.labeling.model import LabelModel
import numpy as np


class PDB:
    """
    Define and train the generative model for the probabilistic decision boundary
    In this class, we utilized the existing functions of following "Snorkel" library:
    https://github.com/snorkel-team/snorkel
    """
    
    def __init__(self, smodels):
        """
        Args:
            smodels: surrogate models to be converted to labeling function
        """
        self.smodels = smodels
        self.gen_model = LabelModel(cardinality=2, verbose=True)

    def label_matrix(self, surrogate_models, data, a):
        """
        Convert the output of the surrogate models to be discrete and aggregate them as a matrix.
        
        Args:
            surrogate_models: surrogate models used in other parts
            data: all available data with or without labels
            a: beta parameter for coverting surrogate models' outputs
        """
        
        def step(y, th):
            y[np.abs(y - 0.5) < th] = -1
            y[y > 0.5] = 1
            y[np.abs(y) < 0.5] = 0
            return y

        matrix = []
        for model in tqdm(surrogate_models):
            try:
                prob = model.predict_proba(data)[:,1]
                pred = step(prob, a)
            except:
                pred = model.predict(data)
            matrix.append(pred)        
        return np.array(matrix)
    
    def fit_all(self, x, beta=0.1):
        """
        Train the generative model with the output of the surrogate models.
        
        Args:
            x: data for training the generative model
            beta: parameter to control the abstain interval
        """
        
        L = self.label_matrix(self.smodels, x, beta)
        self.gen_model.fit(L.T, n_epochs=500, log_freq=50, seed=25)
        
    def predict(self, x, beta=0.1):
        """
        Predict the discrete label.
        
        Args:
            x: data to predict the discrete label
            beta: parameter to control the abstain interval
        """
        
        L = self.label_matrix(self.smodels, x, beta)
        return self.gen_model.predict(L=L.T, tie_break_policy="abstain")
    
    def predict_proba(self, x, beta=0.1):
        """
        Predict the labeling probability.
        
        Args:
            x: data to predict the labeling probability
            beta: parameter to control the abstain interval
        """
        
        L = self.label_matrix(self.smodels, x, beta)
        return self.gen_model.predict_proba(L=L.T)
    
#     def adaptive_threshold(self, x, y)
        


