import pandas as pd
import numpy as np
import random
from pathlib import Path
from CausalGraphicalModel import CausalGraphicalModel
import pyAgrum.causal as csl
pd.option_context('display.max_rows', None)


def bootstrap_test(tot_samples: int) -> None:
    '''
    The method generates a sample of training datasets, each resampled (with replacemnt) using 
    selection probabilities proportional to the survey weight variable "tsWghtP" but using a different
    random seed every time. Then it uses these datasets to estimate the corresponding effect 
    on EV ownership (Y="Already own electric car/van") and saves these into a csv for further statistical analysis.
    '''
    base_path = Path(r'C:/Causal_barriers-EV_uptake_local_code/Causal_barriers-EV_uptake/DATA')
    csv_path = base_path / 'unweighted_dataset.csv'
        
    original_dataset = pd.read_csv(filepath_or_buffer=csv_path, sep=",")
    #reweighted_dataset = original_dataset.copy(deep=True)
    
    bootstrap_treatmt_reslt = pd.DataFrame({'Random_seed': pd.Series(dtype='int'), 
                                    'TE_bootstrap "Already own electric car/van" (pp)': pd.Series(dtype='float'),
                                    'TE_bootstrapNOtConsider "Not considering to buy one" (pp)': pd.Series(dtype='float')})
    
    
    for random_seed in range(1, tot_samples):
        print(random_seed)
        reweighted_dataset = original_dataset.sample(n=int(len(original_dataset)), weights='tsWghtP_n', random_state=random_seed, axis=0, replace=True)
        reweighted_dataset.drop(columns=['tsWghtP_n'], inplace=True)
        
        #print(reweighted_dataset)
        
        causal_grap_model = CausalGraphicalModel(dataset_name='processed_dataset.csv')
        causal_grap_model.build(for_refutation=True, dataset_for_refutation=reweighted_dataset)
        cm = causal_grap_model.c_model # the pyAgrum CausalModel object
        
        _, potential, _ = csl.causalImpact(cm, on='Y', doing='V_7', knowing={})
        
        arr = potential.toarray()
        first_col = arr[:, 0]
        forth_col = arr[:, 3]
        
        TE_bootstrap = float((first_col[0]-first_col[1])*100) # the change in probability of the state "Already own electric car/van", measured in percentage points
        TE_bootstrapNOtConsider = float((forth_col[0]-forth_col[1])*100) # the change in probability of the state "Not considering to buy one", measured in percentage points
        
        new_row = {'Random_seed': random_seed, 
                   'TE_bootstrap "Already own electric car/van" (pp)': TE_bootstrap,
                   'TE_bootstrapNOtConsider "Not considering to buy one" (pp)': TE_bootstrapNOtConsider}
        bootstrap_treatmt_reslt = pd.concat([bootstrap_treatmt_reslt, pd.DataFrame([new_row])], ignore_index=True)
        bootstrap_treatmt_reslt.to_csv(path_or_buf="DATA/REFUTATION_TEST_RESULTS/bootstrap_treatment_results.csv", index=False)
    
        print(f'Random seed {random_seed} out of {tot_samples}. TE_bootstrap "Already own electric car/van" = {TE_bootstrap} (pp)')
        
        
        
    return


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
                                          'TE_placebo "Already own electric car/van" (pp)': pd.Series(dtype='float'),
                                          'TE_placeboNOtConsider "Not considering to buy one" (pp)': pd.Series(dtype='float')})
    
    
    for random_seed in range(1, tot_samples):
        np.random.seed(random_seed)
        shuffled_dataset['V_7'] = np.random.permutation(shuffled_dataset['V_7'].values)
        
        causal_grap_model = CausalGraphicalModel(dataset_name='processed_dataset.csv')
        causal_grap_model.build(for_refutation=True, dataset_for_refutation=shuffled_dataset)
        cm = causal_grap_model.c_model # the pyAgrum CausalModel object
        
        _, potential, _ = csl.causalImpact(cm, on='Y', doing='V_7', knowing={})
    
        arr = potential.toarray()
        first_col = arr[:, 0]
        forth_col = arr[:, 3]
        
        TE_placebo = float((first_col[0]-first_col[1])*100) # the change in probability of the state "Already own electric car/van", measured in percentage points
        TE_placeboNOtConsider = float((forth_col[0]-forth_col[1])*100) # the change in probability of the state "Not considering to buy one", measured in percentage points
        
        new_row = {'Random_seed': random_seed, 
                   'TE_placebo "Already own electric car/van" (pp)': TE_placebo,
                   'TE_placeboNOtConsider "Not considering to buy one" (pp)': TE_placeboNOtConsider}
        plecebo_treatmt_reslt = pd.concat([plecebo_treatmt_reslt, pd.DataFrame([new_row])], ignore_index=True)
        plecebo_treatmt_reslt.to_csv(path_or_buf="DATA/REFUTATION_TEST_RESULTS/placebo_treatment_results.csv", index=False)
    
        print(f'Random seed {random_seed} out of {tot_samples}. TE_placebo "Already own electric car/van" = {TE_placebo} (pp)')
        
    
    return




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
                                          'TE_subsample "Already own electric car/van" (pp)': pd.Series(dtype='float'),
                                          'TE_subsampleNOtConsider "Not considering to buy one" (pp)': pd.Series(dtype='float')})
    
    
    subsample_size = 0.8 # fraction of the original dataset    
    
    for random_seed in range(1, tot_samples):
        subsample_dataset = original_dataset.sample(frac=subsample_size, random_state=random_seed) 
        
        causal_grap_model = CausalGraphicalModel(dataset_name='processed_dataset.csv')
        causal_grap_model.build(for_refutation=True, dataset_for_refutation=subsample_dataset)
        cm = causal_grap_model.c_model # the pyAgrum CausalModel object
        
        _, potential, _ = csl.causalImpact(cm, on='Y', doing='V_7', knowing={})
    
        arr = potential.toarray()
        first_col = arr[:, 0]
        forth_col = arr[:, 3]
    
        TE_subsample = float((first_col[0]-first_col[1])*100) # the change in probability of the state "Already own electric car/van", measured in percentage points
        TE_subsampleNOtConsider = float((forth_col[0]-forth_col[1])*100) # the change in probability of the state "Not considering to buy one", measured in percentage points
        
        new_row = {'Random_seed': random_seed, 
                   'TE_subsample "Already own electric car/van" (pp)': TE_subsample, 
                   'TE_subsampleNOtConsider "Not considering to buy one" (pp)': TE_subsampleNOtConsider}
        data_subsample_reslt = pd.concat([data_subsample_reslt, pd.DataFrame([new_row])], ignore_index=True)
        data_subsample_reslt.to_csv(path_or_buf="DATA/REFUTATION_TEST_RESULTS/subsample_treatment_results.csv", index=False)
    
        print(f'Random seed {random_seed} out of {tot_samples}. TE_subsample "Already own electric car/van" = {TE_subsample} (pp)')


def check_overlap() -> None:

    base_path = Path(
        r"C:/Causal_barriers-EV_uptake_local_code/Causal_barriers-EV_uptake/DATA"
    )
    csv_path = base_path / "unweighted_dataset.csv"

    original_dataset = pd.read_csv(filepath_or_buffer=csv_path, sep=",")

    col_v7 = "V_7"
    col_v8 = "V_8"
    col_v9 = "V_9"

    # Ensure V8 and V9 are treated as categorical to retain unobserved combinations
    original_dataset[col_v8] = original_dataset[col_v8].astype("category")
    original_dataset[col_v9] = original_dataset[col_v9].astype("category")

    total_households = len(original_dataset)
    total_theoretical_strata = 42

    # Explicitly set observed=False to count unobserved strata without warnings
    grouped = original_dataset.groupby([col_v8, col_v9], observed=False)
    
    # Explicitly set observed=False here as well to silence the FutureWarning
    total_observed_strata = (
        original_dataset.groupby([col_v8, col_v9], observed=False)[col_v7].count() > 0
    ).sum()

    failing_positivity_strata = 0
    failing_positivity_households = 0
    overlapping_strata_keys = []

    # Iterate through each stratum to check V7 coverage
    for stratum_key, stratum_df in grouped:
        unique_v7_count = stratum_df[col_v7].nunique()
        stratum_size = len(stratum_df)

        # Violates positivity if fewer than 2 parking categories are present (0 or 1)
        if unique_v7_count < 2:
            failing_positivity_strata += 1
            failing_positivity_households += stratum_size
        else:
            overlapping_strata_keys.append(stratum_key)

    pct_failing_hh = (failing_positivity_households / total_households) * 100

    print("=== EMPIRICAL POSITIVITY & OVERLAP SUMMARY ===")
    print(f"Joint domain size (|V8| x |V9|): {total_theoretical_strata} theoretical strata")
    print(f"[X] Total strata failing positivity (< 2 parking categories): {failing_positivity_strata} (out of {total_theoretical_strata})")
    print(f"[Y%] Household units in non-overlapping strata: {failing_positivity_households} ({pct_failing_hh:.2f}%)\n")

    return
    

if __name__ == "__main__":
    #placebo_treatment_test(tot_samples=1000)
    #data_subsample_test(tot_samples=1000)
    #bootstrap_test(tot_samples=1000)
    
    check_overlap()
    
    
    
 