from CausalGraphicalModel import CausalGraphicalModel
import pyAgrum.causal as csl
import pyAgrum as gum
import numpy as np
import pandas as pd

causal_grap_model = CausalGraphicalModel(dataset_name='processed_dataset.csv')
causal_grap_model.build()



cm = causal_grap_model.c_model # the pyAgrum CausalModel object


formula, potential, explanation = csl.causalImpact(
    cm, 
    on='Y', 
    doing='V_1',
    knowing={}
)

# 1. Get the LaTeX formula
latex_str = formula.toLatex()
print(f"LaTeX Formula: {latex_str}")

# 2. Access the Potential (the numerical distribution)
print("Resulting Potential:")
print(potential)


# 3. See how it was identified 
print(f"Method: {explanation}")





ve = gum.VariableElimination(causal_grap_model.b_net)
p_V_1 = ve.evidenceJointImpact(targets=['V_1'], evs={}) #returns a pyAgrum.Potential for P(targets|evs) for all instantiations (values) of targets and evs variables.

print(p_V_1)



#-----------------------------------------------------
def _weighted_avg(pot, weights):
    numerator = np.sum(pot * weights[:, np.newaxis], axis=0)
    denominator = np.sum(weights)
    return numerator / denominator

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
    index=['Low Income (V_1 vals: 1 to 6)', 'High Income (V_1 vals: 7 to 8)'],
    columns=['Y val=1', 'Y val=2', 'Y val=3', 'Y val=4']
)

print("2-fold Post-treatment distribution P(Y | do(V_1)):")
print("______________________________________________________")
print(final_results)