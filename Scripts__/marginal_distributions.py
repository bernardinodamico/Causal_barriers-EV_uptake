from CausalGraphicalModel import CausalGraphicalModel
import pyAgrum as gum
from Values_mapping import GetVariableValues


causal_grap_model = CausalGraphicalModel(dataset_name='processed_dataset.csv')
causal_grap_model.build()

ve = gum.VariableElimination(causal_grap_model.b_net)

variable_set = causal_grap_model.b_net.names()

for var in variable_set:
    var_potential = ve.evidenceJointImpact(targets=[var], evs={})
    #print(var_potential)
    var_arr_potential = var_potential.toarray()
    
    #var_states = GetVariableValues.get_nums(var_symbol=var)
    var_states = GetVariableValues.get_labels(var_symbol=var)
    
    print(f"MARGINAL P({var}):")
    for i in range(0, len(var_states)):
        string = f"{i} - {var_states[i]}; P({i}) = {round(var_arr_potential[i], 3)}"
        print(string)
    print("------------------")
    

    