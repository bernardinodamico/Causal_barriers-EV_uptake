import pandas as pd
import numpy as np
import random
pd.option_context('display.max_rows', None)



def placebo_treatment_test(tot_samples: int) -> None:
    '''
    The method generates a sample of training datasets where the values of the treatment variable V_7 are randomly 
    shuffled (placebo tretment). Then it uses these datasets to estimate the corresponding effect 
    on EV ownership (Y="Already own electric car/van") and saves these into a csv for further statistical analysis.
    '''
    
    
    return