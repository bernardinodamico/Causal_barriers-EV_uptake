import pyAgrum as gum
import pyAgrum.lib.image as gimg
import pandas as pd
import os

'''
Script to learn the partially directed graph (a Maximal Ancestral Graphs, MAG) form the data. 
It uses the Multivariate Information-based Inductive Causation (MIIC) alghoritm, developed by the Isambert Lab at the Curie institute.
NOTE: the outputtted MAG includes undirected edges, hence it requires manual refinement to run causal identification/estimation.
'''

 
csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DATA", "processed_dataset_reduced_folds.csv")
dataset = pd.read_csv(filepath_or_buffer=csv_path, sep=",")

learner = gum.BNLearner(dataset)

learner.useMIIC() 
learner.useNMLCorrection() # Normalized Maximum Likelihood (NML) correction

# Structural constraints (Prior domain knowledge)
learner.addForbiddenArc('Y', 'V_2')
learner.addForbiddenArc('Y', 'V_13')
learner.addForbiddenArc('V_2', 'V_7')
learner.addForbiddenArc('V_8', 'V_9')
learner.addForbiddenArc('V_7', 'V_9')
learner.addForbiddenArc('V_7', 'V_13')
learner.addForbiddenArc('V_8', 'V_14')
learner.addForbiddenArc('V_14', 'V_8')
learner.addForbiddenArc('V_2', 'V_8')
learner.addForbiddenArc('V_7', 'V_8')
learner.addForbiddenArc('V_7', 'V_14')

# Mandatory edges
learner.addMandatoryArc('V_7', 'Y')
learner.addMandatoryArc('V_1', 'Y')
learner.addMandatoryArc('V_1', 'V_8')
learner.addMandatoryArc('V_1', 'V_13')
#-------------------------------------

learnt_mixed_graph = learner.learnPDAG()




print(" ")
print("--- CAUSAL DISCOVERY RESULTS ---")

# Directed Arcs (->)
for tail, head in learnt_mixed_graph.arcs():
    mi = learner.correctedMutualInformation(tail, head)
    print(f"Directed edge (Arc): {dataset.columns[tail]} --> {dataset.columns[head]}")

# Undirected Edges (-)
for u, v in learnt_mixed_graph.edges():
    print(f"Undirected edge: {dataset.columns[u]} --- {dataset.columns[v]}")
    
# Bidirected Arcs (<->) representing Latent Common Causes identified via MIIC
latents = learner.latentVariables()
if latents:
    print(f"\nDetected {len(latents)} Latent Variable(s):")
    for lat in latents:
        u, v = lat[0], lat[1]
        print(f"BIDIRECTIONAL (Latent): {dataset.columns[u]} <--> {dataset.columns[v]}")
else:
    print("\nNo Bidirected Arcs (Latents) detected.")
    
    
