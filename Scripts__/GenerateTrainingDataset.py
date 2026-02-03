import pandas as pd
import time
import datetime
from DataFusion import DataFusion


'''
Below is a function instantiating the DataFusion class. It builds the dataset 
for training the model parameters (and learning the graph).
'''

def gen_training_dataset(weighted_resampling: bool, parking_prov_folds: str, tenure_folds: str, sample_multiplier: float, work_folds: str, dataset_name: str):
    
    '''
    param resampling: if true, the dataset is resampled using weights
    so to be representative of the Scotland's households population
    The final sample size of the resampled dataset is equal to the original
    size times the multiplier
    '''
    
    start_time = time.time()
    combined_ds_obsrv_vars = pd.DataFrame()

    
    dp = DataFusion(subset_only=False, how_many=200)
    dp.clean_up()
    dp.parking_provision_classification(classification=parking_prov_folds)
    dp.working_status_classification(classification=work_folds)
    dp.tenure_classification(classification=tenure_folds)
    dp.fill_in_infrastruct_density()
    if weighted_resampling is True:
        dp.weighted_resampling(sample_size=int(len(dp.ds_obsrv_vars) * sample_multiplier))
    dp.rename_reorder_vars()
    
    combined_ds_obsrv_vars = dp.ds_obsrv_vars
    
    
    # save processed datasets to csv files   
    combined_ds_obsrv_vars.to_csv(path_or_buf=f"DATA/{dataset_name}.csv", index=False)
    
    end_time = time.time()

    print("Data pre-processing time (h:m:s) ", str(datetime.timedelta(seconds = round(end_time - start_time, 0))))

    return combined_ds_obsrv_vars








if __name__ == "__main__":
    gen_training_dataset(weighted_resampling=True, 
                         sample_multiplier=1.0, 
                         parking_prov_folds='2-fold', 
                         tenure_folds='3-fold', 
                         work_folds='2-fold',
                         dataset_name='processed_dataset_reduced_folds'
                         )