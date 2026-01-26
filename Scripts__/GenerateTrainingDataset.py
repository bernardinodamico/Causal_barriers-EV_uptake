import pandas as pd
import time
import datetime
from DataFusion import DataFusion


'''
Below is a function instantiating the DataFusion class. It builds the dataset 
for training the model parameters (and learning the graph).
'''

def gen_training_dataset():
    
    start_time = time.time()
    combined_ds_obsrv_vars = pd.DataFrame()

    
    dp = DataFusion(subset_only=False, how_many=200)
    dp.clean_up()
    dp.parking_provision_classification(classification='2-fold')
    dp.working_status_classification()
    dp.tenure_classification()
    dp.fill_in_infrastruct_density()
    dp.weighted_resampling(sample_size=len(dp.ds_obsrv_vars)*20)
    dp.rename_reorder_vars()
    
    combined_ds_obsrv_vars = dp.ds_obsrv_vars
    
    
    # save processed datasets to csv files   
    combined_ds_obsrv_vars.to_csv(path_or_buf="DATA/processed_dataset.csv", index=False)
    
    end_time = time.time()

    print("Data pre-processing time (h:m:s) ", str(datetime.timedelta(seconds = round(end_time - start_time, 0))))

    return combined_ds_obsrv_vars








if __name__ == "__main__":
    gen_training_dataset()