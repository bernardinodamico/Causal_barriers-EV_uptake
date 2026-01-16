import pandas as pd
from pandas import DataFrame
import os
import math
import numpy as np
#from Values_mapping import GetVariableValues
import time
import datetime



class DataFusion():

    ds_obsrv_vars: DataFrame = None


    def __init__(self, subset_only: bool = False, how_many: int = 100):

        self.initialise_dset_obsrv_vars(subset_only=subset_only, how_many=how_many)

        return
    
    
    def initialise_dset_obsrv_vars(self, subset_only: bool = False, how_many: int = 100) -> None:
        
        self.ds_obsrv_vars = pd.DataFrame()
        
        scottish_household_survey_ds = pd.read_excel(io=os.path.join(os.path.dirname(__file__), r"DATA\RAW\Scottish_Household_Survey_2022.xlsx"), sheet_name=f"shs2022_social_public")
        
        self.ds_obsrv_vars.loc[:, 'Y'] = scottish_household_survey_ds.loc[:, 'gpawr2c'] # whether random adult in household considers buying (or already has) a plug-in electric car or van
        self.ds_obsrv_vars.loc[:, 'V_1'] = scottish_household_survey_ds.loc[:, 'tothinc'] # annual net household income [£]
        
        if subset_only is True:
            self.ds_obsrv_vars = self.ds_obsrv_vars[:how_many] # only keep first n rows (n=how_many)
            
            
        return
    
    
    


 
    
'''
Below is a function instantiating the DataFusion class. It builds the dataset 
for training the model parameters of the Causal Bayesian Network.
'''

def gen_training_dataset():
    
    start_time = time.time()
    combined_ds_obsrv_vars = pd.DataFrame()

    
    dp = DataFusion(subset_only=True, how_many=200)
    
    combined_ds_obsrv_vars = dp.ds_obsrv_vars
    
    
    # save processed datasets to csv files   
    combined_ds_obsrv_vars.to_csv(path_or_buf="DATA/processed_dataset.csv", index=False)
    
    end_time = time.time()

    print("Data pre-processing time (h:m:s) ", str(datetime.timedelta(seconds = round(end_time - start_time, 0))))

    return combined_ds_obsrv_vars




gen_training_dataset()
