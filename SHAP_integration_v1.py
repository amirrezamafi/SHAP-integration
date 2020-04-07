
import pandas as pd
import numpy as np
import shap

shap.initjs()



def calculate_shap1 (model , data):
    
    '''
        The input data is a dataframe, consists of only the model`s features columns
        The shap values by default are toward class_1
        
        
        Parameters
        ----------
        model: the machine learning model which has been used (ex: RandomForest)
            
        data: the dataset which its Shap values is required. data is a dataframe, consists of only the 
        features columns
        
        '''
    
    explainer = shap.TreeExplainer(model)
    shap_values_df = pd.DataFrame(explainer.shap_values(data)[1], columns=((data.columns)))
    
    return shap_values_df

    '''
        The function calcule_shap returns a Pandas DataFrame
        in shape of (number of instance in data , number of features)
        
        '''


def calculate_shap2 (model , data, features):
    
    '''
        The data is a dataframe, consists of all data`s columns (not only the model`s features)
        Features are specified by a list 
        The shap values by default are toward class_1
        
        Parameters
        ----------
        model: the machine learning model which has been used (ex: RandomForest)
            
        data: the dataset which its Shap values is required
        
        features: the list of the features of the model
        
        '''
    
    explainer = shap.TreeExplainer(model)
    shap_values_df = pd.DataFrame(explainer.shap_values(data[features])[1], columns=(features))
    return shap_values_df

    '''
        The function calcule_shap returns a Pandas DataFrame
        in shape of (number of instance in data , number of features)
        
        '''    



def calculate_shap3 (model , data, features, to_class):
    
    '''
        The data is a dataframe, consists of all data`s columns (not only the model`s features)
        Features are specified by a list
        The direction of the shap values (toward class_0 or class_1) must be specified
        
        Parameters
        ----------
        model: the machine learning model which has been used (ex: RandomForest)
            
        data: the dataset which its Shap values is required
        
        features: the list of the features of the model
        
        to_class: the class which the Shap values toward it, is required (to_class=0: the output will be the 
        Shap values toward class 0; to_class=1: the output will be the Shap values toward class 1)
        
        '''
    
    explainer = shap.TreeExplainer(model)
    shap_values_df = pd.DataFrame(explainer.shap_values(data[features])[to_class], columns=(features))
    return shap_values_df

    '''
        The function calcule_shap returns a Pandas DataFrame
        in shape of (number of instance in data , number of features)
        
        '''    

