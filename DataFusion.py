import pandas as pd
from pandas import DataFrame
import os
import numpy as np
from Values_mapping import VariableValues



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
            'D10', # 'V_7' parking provision 
            'typdwell', # 'V_8' type of dwelling
            'C5', # 'V_9' Date of construction of dwelling
            'tsWghtP_n' 
            # "tsWghtP_n" is the normalised paired weight used to scale the sub-sample 
            # of households -- that completed both the social interview (SHS) and physical 
            # inspection (SHCS) -- to be representative of the national Scottish population.
        ]
        
        household_variables = [
            'UNIQIDNEW',
            'gpawr2c', # whether random adult in household considers buying (or already has) a plug-in electric car or van
            'tothinc', # annual net household income [£]
            'NUMCARS_NEW', # number of cars available for private use by members of household
            'totads', # total number of adult people in the household
            'totkids', # total number of children in the household
            'hhtype_new', #  household composition
            'SHS_2CLA', #  household urban/rural classification
            'council', #  local authority where the household resides
            'hhwork', #  Hosehold working status
            'tenure_harm', #  tenure
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
    
    
    def clean_up(self) -> None:
        self.ds_obsrv_vars = self.ds_obsrv_vars.dropna() # removes rows with empy cells
        self.ds_obsrv_vars = self.ds_obsrv_vars.drop(columns=['UNIQIDNEW', 'uniqidnew_shs_social']) # removes ID columns
        
        self.ds_obsrv_vars = (self.ds_obsrv_vars.loc[self.ds_obsrv_vars['D10'] != 7].reset_index(drop=True)) # removes rows where D10 = 7
        self.ds_obsrv_vars = (self.ds_obsrv_vars.loc[self.ds_obsrv_vars['D10'] != 8].reset_index(drop=True)) # removes rows where D10 = 8
        self.ds_obsrv_vars = (self.ds_obsrv_vars.loc[self.ds_obsrv_vars['D10'] != 9].reset_index(drop=True)) # removes rows where D10 = 9
        
        self.ds_obsrv_vars = (self.ds_obsrv_vars.loc[self.ds_obsrv_vars['C5'] != 9].reset_index(drop=True)) # removes rows where C5 = 9
        
        self.ds_obsrv_vars['D10'] = self.ds_obsrv_vars['D10'].astype(int)
        self.ds_obsrv_vars['gpawr2c'] = self.ds_obsrv_vars['gpawr2c'].astype(int)
        self.ds_obsrv_vars['tothinc'] = self.ds_obsrv_vars['tothinc'].astype(int)
        
        self.ds_obsrv_vars = (self.ds_obsrv_vars.loc[self.ds_obsrv_vars['gpawr2c'] != 6].reset_index(drop=True)) # removes rows where gpawr2c = 6
        
        return


    def working_status_classification(self, classification: str) -> None:
        if classification == '5-fold':
            pass
        elif classification == '2-fold':
            conditions = [
                (self.ds_obsrv_vars['hhwork'].isin([1, 3, 4])), # One or more working adults
                (self.ds_obsrv_vars['hhwork'].isin([2, 5]))     # None working
                ]
            choices = ['1', '2']
            self.ds_obsrv_vars['hhwork'] = np.select(conditions, choices, default=pd.NA)
            self.ds_obsrv_vars['hhwork'] = self.ds_obsrv_vars['hhwork'].astype(int)
        else:
            raise ValueError(f"Invalid argument for classification = '{classification}'. Expected: '5-fold' or '2-fold'.") 
        return


    def tenure_classification(self, classification: str) -> None:
        if classification == '5-fold':
            pass
        elif classification == '3-fold':
            conditions = [
                (self.ds_obsrv_vars['tenure_harm'].isin([1, 2])),  # Owned (outright or mortgage)
                (self.ds_obsrv_vars['tenure_harm'].isin([3])),     # Part mortgage, part rent
                (self.ds_obsrv_vars['tenure_harm'].isin([4, 5]))      # Rented (LA, Co-op, private landlord)
                ]
            choices = ['1', '2', '3']
            self.ds_obsrv_vars['tenure_harm'] = np.select(conditions, choices, default=pd.NA)
            self.ds_obsrv_vars['tenure_harm'] = self.ds_obsrv_vars['tenure_harm'].astype(int)
        else:
            raise ValueError(f"Invalid argument for classification = '{classification}'. Expected: '5-fold' or '3-fold'.") 
        return


    def parking_provision_classification(self, classification: str) -> None:
        if classification == '6-fold':
            pass
        elif classification == '2-fold':
            conditions = [
            (self.ds_obsrv_vars['D10'].isin([1, 2, 3, 4])), # off_street
            (self.ds_obsrv_vars['D10'].isin([5, 6]))        # on_street
            ]
            choices = ['1', '2']
            
            self.ds_obsrv_vars['D10'] = np.select(conditions, choices, default=pd.NA)
            self.ds_obsrv_vars['D10'] = self.ds_obsrv_vars['D10'].astype(int)
        else:
            raise ValueError(f"Invalid argument for classification = '{classification}'. Expected: '6-fold' or '2-fold'.")    
        return


    def _compute_infrastruct_densities(self) -> tuple[dict, dict]:
        
        Workplace_charging_ds = pd.read_excel(io=os.path.join(os.path.dirname(__file__), r"DATA\RAW\electric-vehicle-charging-device-grant-scheme-statistics-january-2023.xlsx"), sheet_name=f"3", header=2)
        Public_charging_ds = pd.read_excel(io=os.path.join(os.path.dirname(__file__), r"DATA\RAW\electric-vehicle-public-charging-infrastructure-statistics-october-2025.xlsx"), sheet_name=f"1a", header=2)
        pop_estimate = pd.read_excel(io=os.path.join(os.path.dirname(__file__), r"DATA\RAW\data-mid-year-population-estimates-2022.xlsx"), sheet_name=f"Table 1", header=3)

        council_pops = []
        for i in range(3, 35):
            council_pops.append(pop_estimate.loc[i, "All ages"])

        council_labels = []
        
        workplace_installs = []
        for i in range(354, 386):
            council_labels.append(Workplace_charging_ds.loc[i, "Region or Local Authority"])
            workplace_installs.append(Workplace_charging_ds.loc[i, "Sockets Installed"])
        
        public_installs = []
        for i in range(389, 421):
            public_installs.append(Public_charging_ds.loc[i, "Jan-23"])
            
        
        vv = VariableValues()
        V_14_vlabels = list(vv.Variables_dic['V_14'].values())
        V_14_vnums = list(vv.Variables_dic['V_14'].keys())
        
        #----------------------------------------------------------------------------------------
        wp_installs = {}
        pb_installs = {}
        
        for i in range(0, len(V_14_vlabels)):
            for j in range(0, len(council_labels)):
                if V_14_vlabels[i] == council_labels[j]:
                    
                    per_capita_workplace_installs = (workplace_installs[j] / council_pops[j]) * 100000
                    wp_installs[V_14_vnums[i]] = per_capita_workplace_installs
                    
                    
                    per_capita_public_install = (public_installs[j] / council_pops[j]) * 100000
                    pb_installs[V_14_vnums[i]] = per_capita_public_install
                    
                    break
                else:
                    continue
        
        return wp_installs, pb_installs



    def fill_in_infrastruct_density(self) -> None:
        wp_installs, pb_installs = self._compute_infrastruct_densities()
        vv = VariableValues()
        
        V_10_values = vv.Variables_dic['V_10']
        bin_edges_wp = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
        discretised_wp_installs = {}
        for k, v in wp_installs.items():
            bin_id_wp = np.digitize(v, bin_edges_wp, right=False)
            bin_id_wp = min(bin_id_wp, len(V_10_values))
            discretised_wp_installs[k] = bin_id_wp # a dict with key = LA value, and dict-value = discretised workplace installs density

        V_11_values = vv.Variables_dic['V_11']
        bin_edges_pb = np.array([0.0, 50.0, 100.0, 150.0, 200.0, 250.0])
        discretised_pb_installs = {}
        for k, v in pb_installs.items():
            bin_id_pb = np.digitize(v, bin_edges_pb, right=False)
            bin_id_pb = min(bin_id_pb, len(V_11_values))
            discretised_pb_installs[k] = bin_id_pb # a dict with key = LA value, and dict-value = discretised public installs density
        #----------------------------------------------------------------------------------------------------------------
        
        self.ds_obsrv_vars["workpl_charging_density"] = (self.ds_obsrv_vars["council"].astype(str).map(discretised_wp_installs)).astype("Int64")
        self.ds_obsrv_vars["public_charging_density"] = (self.ds_obsrv_vars["council"].astype(str).map(discretised_pb_installs)).astype("Int64")
        
        return
    
    
    
    def weighted_resampling(self, sample_size: int) -> None:
        '''
        The method uses the 'tsWghtP_n' weight value (in the SHCS) to generate a re-sampled dataset by drawing from
        the existing dataset based on the weight values assignded to each unit.
        By doing so, the returned sample dataset is representative of the Scottish household population
        - sample_size parameter is the total number of draws
        '''
        sampled_df = self.ds_obsrv_vars.sample(n=sample_size, weights='tsWghtP_n', random_state=42, axis=0, replace=True)

        self.ds_obsrv_vars = sampled_df 
        return
   
    
    def rename_reorder_vars(self) -> None:
        
        #self.ds_obsrv_vars.drop(columns=['council'], inplace=True)
        
        column_name_mapping = {
            'gpawr2c': 'Y',
            'tothinc': 'V_1',
            'NUMCARS_NEW': 'V_2',
            'totads': 'V_3',
            'totkids': 'V_4',
            'hhtype_new': 'V_5',
            'SHS_2CLA': 'V_6',
            'D10': 'V_7',
            'typdwell': 'V_8',
            'C5': 'V_9',
            'workpl_charging_density': 'V_10',
            'public_charging_density': 'V_11',
            'hhwork': 'V_12',
            'tenure_harm': 'V_13',
            'council': 'V_14'
        }
        
        self.ds_obsrv_vars.rename(columns=column_name_mapping, inplace=True)
        self.ds_obsrv_vars.drop(columns=['tsWghtP_n'], inplace=True)
        
        
        self.ds_obsrv_vars = self.ds_obsrv_vars[['Y', 'V_1', 'V_2', 'V_3', 'V_4', 'V_5', 'V_6', 'V_7', 'V_8', 'V_9', 'V_10', 'V_11', 'V_12', 'V_13', 'V_14']]
        
        return
 
    








