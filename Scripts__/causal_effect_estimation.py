from CausalGraphicalModel import CausalGraphicalModel
import pyAgrum.causal as csl


causal_grap_model = CausalGraphicalModel()
causal_grap_model.build()



cm = causal_grap_model.c_model # the pyAgrum CausalModel object


formula, potential, explanation = csl.causalImpact(
    cm, 
    on='Y', 
    doing='V_7'
)

# 1. Get the LaTeX formula
latex_str = formula.toLatex()
print(f"LaTeX Formula: {latex_str}")

# 2. Access the Potential (the numerical distribution)
print("Resulting Potential:")
print(potential)

# 3. See how it was identified 
print(f"Method: {explanation}")