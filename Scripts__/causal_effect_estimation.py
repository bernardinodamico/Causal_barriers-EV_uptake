from CausalGraphicalModel import CausalGraphicalModel
import pyAgrum.causal as csl
import pyAgrum as gum
import numpy as np
import pandas as pd
from pyAgrum.causal import CausalModel

def compute_effect_of_ParkingProv_on_EV(cm: CausalModel) -> None:
    formula, potential, explanation = csl.causalImpact(cm, on='Y', doing='V_7', knowing={})
    
    # printing out
    print("Post-treatment distribution P(Y | do(Parking prov.)):")
    print(potential)
    print(f"Method: {explanation}\n")
    print(f"LaTeX Formula:\n {formula.toLatex()}")
    print("_____________________________________________________________________")
    return

def compute_effect_of_Income_on_EV(cm: CausalModel) -> None:
    formula, potential, explanation = csl.causalImpact(cm, on='Y', doing='V_1', knowing={})

    '''
    Note: all the below code is just to convert the identified 8-fold post-tretment 
    potential into a 2-fold (low-income; high-income) potential
    '''
    
    ve = gum.VariableElimination(causal_grap_model.b_net)
    p_V_1 = ve.evidenceJointImpact(targets=['V_1'], evs={}) #returns a pyAgrum.Potential for P(targets|evs) for all instantiations (values) of targets and evs variables.

    #-----------------------------------------------------
    def _weighted_avg(pot, weights):
        numerator = np.sum(pot * weights[:, np.newaxis], axis=0)
        denominator = np.sum(weights)
        return numerator / denominator
    #-----------------------------------------------------

    arr_Prob_Y_do_V1 = potential.toarray()
    arr_Prob_V_1 = p_V_1.toarray()

    # Slice for Low Income (Bands 1-6 -> indices 0,1,2,3,4,5)
    low_pot = arr_Prob_Y_do_V1[0:6, :]
    low_weights = arr_Prob_V_1[0:6]

    # Slice for High Income (Bands 7-8 -> indices 6,7)
    high_pot = arr_Prob_Y_do_V1[6:8, :]
    high_weights = arr_Prob_V_1[6:8]

    low_dist = _weighted_avg(pot=low_pot, weights=low_weights)
    high_dist = _weighted_avg(pot=high_pot, weights=high_weights)

    final_results = pd.DataFrame(
        [low_dist, high_dist], 
        index=['LowInc', 'HighInc'],
        columns=['1', '2', '3', '4']
    )

    # Convert the binary income dataframe into a PyAgrum potential (just for printing purpose) 
    y_var = gum.LabelizedVariable('Y', 'Y', ['1', '2', '3', '4'])
    V_1_var = gum.LabelizedVariable('V_1', 'V_1', ['LowInc', 'HighInc'])
    
    binary_V_1_pot = gum.Potential()
    
    binary_V_1_pot.add(y_var)
    binary_V_1_pot.add(V_1_var)
    
    for Y_val in range(1, 5):
        binary_V_1_pot[{'V_1': 'LowInc', 'Y': str(Y_val)}] = final_results.loc['LowInc', str(Y_val)]
    for Y_val in range(1, 5):
        binary_V_1_pot[{'V_1': 'HighInc', 'Y': str(Y_val)}] = final_results.loc['HighInc', str(Y_val)]
    
    # printing out
    print("\n2-fold Post-treatment distribution P(Y | do(Income)):")
    print(binary_V_1_pot)
    print(f"Method: {explanation}\n")
    print(f"LaTeX Formula:\n {formula.toLatex()}\n")
    
    return





if __name__ == "__main__":
    causal_grap_model = CausalGraphicalModel(dataset_name='processed_dataset.csv')
    causal_grap_model.build()
    cm = causal_grap_model.c_model # the pyAgrum CausalModel object
    
    compute_effect_of_ParkingProv_on_EV(cm=cm)
    compute_effect_of_Income_on_EV(cm=cm)