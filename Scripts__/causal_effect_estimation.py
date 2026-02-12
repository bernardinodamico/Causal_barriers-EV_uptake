from CausalGraphicalModel import CausalGraphicalModel
import pyAgrum.causal as csl
import pyAgrum as gum

causal_grap_model = CausalGraphicalModel(dataset_name='processed_dataset.csv')
causal_grap_model.build()



cm = causal_grap_model.c_model # the pyAgrum CausalModel object


formula, potential, explanation = csl.causalImpact(
    cm, 
    on='Y', 
    doing='V_7',
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


#--------------------------------------------

#ve = gum.VariableElimination(causal_grap_model.b_net)
#p_Y_given_V_7 = ve.evidenceJointImpact(targets=['Y'], evs={'V_7'}) #returns a pyAgrum.Potential for P(targets|evs) for all instantiations (values) of targets and evs variables.

#print(p_Y_given_V_7)
