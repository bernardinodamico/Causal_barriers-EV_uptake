from CausalGraphicalModel import CausalGraphicalModel
import pyAgrum.causal as csl
import pyAgrum as gum
import numpy as np
import pandas as pd
from pyAgrum.causal import CausalModel

'''
Script to:
- evaluate independency checks (using d-separartion) 
- count undirected simple backdoor paths between variables 
'''


causal_grap_model = CausalGraphicalModel(dataset_name='processed_dataset.csv')
causal_grap_model.build()
causal_grap_model._define_graph()



#--------------------------------------------------------------------------------
G_V_1_underscored = causal_grap_model.get_subgraph(which_graph='G_V_1_underscored') # the original graph "G" after removing all arrows going out of V_1

# count back-door paths between V_1 and Y
causal_grap_model.get_paths(sub_graph=G_V_1_underscored, st_var='V_1', end_var='Y', print_paths=True, print_count=True)

# check if V_1 and Y are conditionally independent (in G_V_1_underscored) given V_5
causal_grap_model.check_independence(graph=G_V_1_underscored, A_nodes={'V_1'}, B_nodes={'Y'}, conditioned_on={'V_5'}, print_res=True)

#--------------------------------------------------------------------------------

G_V_7_underscored = causal_grap_model.get_subgraph(which_graph='G_V_7_underscored') # the original graph "G" after removing all arrows going out of V_7

# count back-door paths between V_7 and Y
causal_grap_model.get_paths(sub_graph=G_V_7_underscored, st_var='V_7', end_var='Y', print_paths=True, print_count=True)

# check if V_7 and Y are conditionally independent (in G_V_7_underscored) given V_9 and V_8
causal_grap_model.check_independence(graph=G_V_7_underscored, A_nodes={'V_7'}, B_nodes={'Y'}, conditioned_on={'V_9', 'V_8'}, print_res=True)
