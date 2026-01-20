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
        scottish_house_condition_ds = pd.read_excel(io=os.path.join(os.path.dirname(__file__), r"DATA\RAW\Scottish_House_Condition_Survey_2022.xlsx"), sheet_name=f"shcs2022_dataset")
        
        house_condition_variables = [
            'uniqidnew_shs_social',
            'D10', # 'V_7' parking provision of the dwelling
            'D1', # 'V_10' entrance level to dwelling
            'typdwell', # 'V_8' type of dwelling
            'C5', # 'V_9' Date of construction of dwelling
        ]
        
        household_variables = [
            'UNIQIDNEW',
            'gpawr2c', # 'Y' whether random adult in household considers buying (or already has) a plug-in electric car or van
            'tothinc', # 'V_1' annual net household income [£]
            'NUMCARS_NEW', # 'V_2' number of cars available for private use by members of household
            'totads', # 'V_3' total number of adult people in the household
            'totkids', # 'V_4' total number of children in the household
            'hhtype_new', # 'V_5' household composition
            'SHS_2CLA', # 'V_6' household urban/rural classification
            #'council', # 'council' local authority where the household resides
        ]
        
        self.ds_obsrv_vars = scottish_house_condition_ds[house_condition_variables].merge(
            scottish_household_survey_ds[household_variables],
            left_on='uniqidnew_shs_social',
            right_on='UNIQIDNEW',
            how='inner'
            )
        
        if subset_only is True:
            self.ds_obsrv_vars = self.ds_obsrv_vars[:how_many] # only keep first n rows (n=how_many)
                 
        return
    
    
    def fill_in_home_charging_var(self) -> None:
        '''
        Add the variable "Home EV charging feasibility" to the dataset, mapping its values based 
        on the parking provision variable (D10) and Entrance level to dwelling (D1)
        '''
        
        conditions = [
            (self.ds_obsrv_vars['D10'].isin([1, 2, 3])),                                                    # feasible
            (self.ds_obsrv_vars['D10'].isin([4])),                                                          # not feasible
            (self.ds_obsrv_vars['D10'].isin([5])) & (self.ds_obsrv_vars['D1'].isin([0, 7])),                # feasible
            (self.ds_obsrv_vars['D10'].isin([5])) & (self.ds_obsrv_vars['D1'].isin([1, 2, 3, 4, 5, 6])),    # not feasible
            (self.ds_obsrv_vars['D10'].isin([6,7]))                                                         # not feasible
        ]
        
        choices = [
                'true',   
                'false', 
                'true',
                'false',  
                'false' 
            ]
        
        self.ds_obsrv_vars['home_charging'] = np.select(conditions, choices, default=pd.NA)
        
        return
    
    
    def clean_up(self) -> None:
        self.ds_obsrv_vars = self.ds_obsrv_vars.dropna() # removes rows with empy cells
        self.ds_obsrv_vars = self.ds_obsrv_vars.drop(columns=['UNIQIDNEW', 'uniqidnew_shs_social']) # removes ID columns
        
        self.ds_obsrv_vars = (self.ds_obsrv_vars.loc[self.ds_obsrv_vars['D10'] != 8].reset_index(drop=True)) # removes rows where D10 = 8
        self.ds_obsrv_vars = (self.ds_obsrv_vars.loc[self.ds_obsrv_vars['D10'] != 9].reset_index(drop=True)) # removes rows where D10 = 9
        
        self.ds_obsrv_vars = (self.ds_obsrv_vars.loc[self.ds_obsrv_vars['D1'] != 8].reset_index(drop=True)) # removes rows where D1 = 8
        self.ds_obsrv_vars = (self.ds_obsrv_vars.loc[self.ds_obsrv_vars['D1'] != 9].reset_index(drop=True)) # removes rows where D1 = 9
        
        self.ds_obsrv_vars = (self.ds_obsrv_vars.loc[self.ds_obsrv_vars['C5'] != 9].reset_index(drop=True)) # removes rows where C5 = 9
        
        self.ds_obsrv_vars = (self.ds_obsrv_vars.loc[self.ds_obsrv_vars['gpawr2c'] != 6.0].reset_index(drop=True)) # removes rows where gpawr2c = 6.0
        self.ds_obsrv_vars = (self.ds_obsrv_vars.loc[self.ds_obsrv_vars['gpawr2c'] != 6].reset_index(drop=True)) # removes rows where gpawr2c = 6
        
        
        return
    


 
    
'''
Below is a function instantiating the DataFusion class. It builds the dataset 
for training the model parameters of the Causal Bayesian Network.
'''

def gen_training_dataset():
    
    start_time = time.time()
    combined_ds_obsrv_vars = pd.DataFrame()

    
    dp = DataFusion(subset_only=False, how_many=200)
    dp.fill_in_home_charging_var()
    dp.clean_up()
    
    combined_ds_obsrv_vars = dp.ds_obsrv_vars
    
    
    # save processed datasets to csv files   
    combined_ds_obsrv_vars.to_csv(path_or_buf="DATA/processed_dataset.csv", index=False)
    
    end_time = time.time()

    print("Data pre-processing time (h:m:s) ", str(datetime.timedelta(seconds = round(end_time - start_time, 0))))

    return combined_ds_obsrv_vars




gen_training_dataset()







