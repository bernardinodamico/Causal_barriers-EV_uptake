import pandas as pd
import numpy as np
import random
from pathlib import Path
from CausalGraphicalModel import CausalGraphicalModel
import pyAgrum.causal as csl
pd.option_context('display.max_rows', None)



def placebo_treatment_test(tot_samples: int) -> None:
    '''
    The method generates a sample of training datasets where the values of the treatment variable V_7 are randomly 
    shuffled (placebo tretment). Then it uses these datasets to estimate the corresponding effect 
    on EV ownership (Y="Already own electric car/van") and saves these into a csv for further statistical analysis.
    '''
    base_path = Path(r'C:/Causal_barriers-EV_uptake_local_code/Causal_barriers-EV_uptake/DATA')
    csv_path = base_path / 'processed_dataset.csv'
        
    original_dataset = pd.read_csv(filepath_or_buffer=csv_path, sep=",")
    shuffled_dataset = original_dataset.copy(deep=True)
    
    
    plecebo_treatmt_reslt = pd.DataFrame({'Random_seed': pd.Series(dtype='int'), 
                                          'TE_placebo "Already own electric car/van" (pp)': pd.Series(dtype='float')})
    
    
    for random_seed in range(1, tot_samples):
        np.random.seed(random_seed)
        shuffled_dataset['V_7'] = np.random.permutation(shuffled_dataset['V_7'].values)
        
        causal_grap_model = CausalGraphicalModel(dataset_name='processed_dataset.csv')
        causal_grap_model.build(for_refutation=True, dataset_for_refutation=shuffled_dataset)
        cm = causal_grap_model.c_model # the pyAgrum CausalModel object
        
        _, potential, _ = csl.causalImpact(cm, on='Y', doing='V_7', knowing={})
    
        arr = potential.toarray()
        first_col = arr[:, 0]
        
        TE_placebo = (first_col[0]-first_col[1])*100 # the change in probability of the state "Already own electric car/van", measured in percentage points
        
        new_row = {'Random_seed': random_seed, 'TE_placebo "Already own electric car/van" (pp)': TE_placebo}
        plecebo_treatmt_reslt = pd.concat([plecebo_treatmt_reslt, pd.DataFrame([new_row])], ignore_index=True)
        plecebo_treatmt_reslt.to_csv(path_or_buf="DATA/REFUTATION_TEST_RESULTS/placebo_treatment_results.csv", index=False)
    
        print(f'Random seed {random_seed} out of {tot_samples}. TE_placebo "Already own electric car/van" = {TE_placebo} (pp)')
        
    
    return


'''
NOTE: fo the subsample test, use the following code:
subsample_size = 0.4 # percentage of the original dataset
    for random_seed in range(1, tot_samples):
        subsample_dataset = subsample_dataset.sample(frac=subsample_size, random_state=random_seed)  
'''


def data_subsample_test(tot_samples: int) -> None:
    '''
    The method generates a sample of training datasets where each dataset is a random subsample of the original training dataset. 
    Then it uses these random subsample datasets to estimate the corresponding effect 
    on EV ownership (Y="Already own electric car/van") and saves these into a csv for further statistical analysis.
    '''
    base_path = Path(r'C:/Causal_barriers-EV_uptake_local_code/Causal_barriers-EV_uptake/DATA')
    csv_path = base_path / 'processed_dataset.csv'
        
    original_dataset = pd.read_csv(filepath_or_buffer=csv_path, sep=",")
    
    data_subsample_reslt = pd.DataFrame({'Random_seed': pd.Series(dtype='int'), 
                                          'TE_subsample "Already own electric car/van" (pp)': pd.Series(dtype='float')})
    
    
    subsample_size = 0.8 # fraction of the original dataset    
    
    for random_seed in range(1, tot_samples):
        subsample_dataset = original_dataset.sample(frac=subsample_size, random_state=random_seed) 
        
        causal_grap_model = CausalGraphicalModel(dataset_name='processed_dataset.csv')
        causal_grap_model.build(for_refutation=True, dataset_for_refutation=subsample_dataset)
        cm = causal_grap_model.c_model # the pyAgrum CausalModel object
        
        _, potential, _ = csl.causalImpact(cm, on='Y', doing='V_7', knowing={})
    
        arr = potential.toarray()
        first_col = arr[:, 0]
    
        TE_subsample = (first_col[0]-first_col[1])*100 # the change in probability of the state "Already own electric car/van", measured in percentage points
        
        new_row = {'Random_seed': random_seed, 'TE_subsample "Already own electric car/van" (pp)': TE_subsample}
        data_subsample_reslt = pd.concat([data_subsample_reslt, pd.DataFrame([new_row])], ignore_index=True)
        data_subsample_reslt.to_csv(path_or_buf="DATA/REFUTATION_TEST_RESULTS/subsample_treatment_results.csv", index=False)
    
        print(f'Random seed {random_seed} out of {tot_samples}. TE_subsample "Already own electric car/van" = {TE_subsample} (pp)')







if __name__ == "__main__":
    #placebo_treatment_test(tot_samples=1000)
    data_subsample_test(tot_samples=1000)