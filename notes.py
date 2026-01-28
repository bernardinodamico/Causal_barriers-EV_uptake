import pyAgrum as gum
import pyAgrum.lib.image as gimg
import pandas as pd
import os












  
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DATA", "processed_dataset.csv")
dataset = pd.read_csv(filepath_or_buffer=csv_path, sep=",")

# 1. Initialize the learner with your dataframe
learner = gum.BNLearner(dataset)


# 3. Choose your discovery algorithm
# MIIC is excellent for causal discovery in pyAgrum
learner.useMIIC() 
learner.useNMLCorrection() # Normalized Maximum Likelihood (NML) correction

# 4. Optional: Add structural constraints (Domain Knowledge)
# We know off-street parking cannot be CAUSED by EV ownership (temporality)
#learner.addForbiddenArc("ev_ownership", "parking_provision")


# Based on prior knowledge, and arch may be forbidden 
# either because it is wrongly oriented cauaslly, or 
# because the captured correlation is to be expained in 
# terms of a confounder (latent or observed).
#for i in range(1, 15):
    # EV ownership cannot cause any other variable in the dataset
    #learner.addForbiddenArc('Y', f'V_{i}')

    
    # Root nodes cannot be caused by other variables in the dataset
    # these nodes below are assumed as root.
    #root_nodes = ['V_3', 'V_4', 'V_9', 'V_10', 'V_13']
    #for r_node in root_nodes:
    #    if i != r_node:
    #        learner.addForbiddenArc(f'V_{i}', r_node)
    #        print(f'forbidden: V_{i} -> {r_node}')


learner.addForbiddenArc('V_2', 'V_3')

learner.addForbiddenArc('V_13', 'V_10')

learner.addForbiddenArc('V_2', 'V_7')

learner.addForbiddenArc('V_5', 'V_4')
learner.addForbiddenArc('V_5', 'V_3')

#learner.addForbiddenArc('V_4', 'V_3')
#learner.addForbiddenArc('V_3', 'V_4')

#learner.addForbiddenArc('V_8', 'V_2')
learner.addForbiddenArc('V_2', 'V_8')

learner.addForbiddenArc('V_7', 'V_8')


# You would force a mandatory arc because of domain 
# knowledge (such as a known physical law or a temporal 
# sequence) that a causal relationship exists.

learner.addMandatoryArc('V_1', 'Y')
#learner.addMandatoryArc('V_1', 'V_2')
learner.addMandatoryArc('V_7', 'Y')
learner.addMandatoryArc('V_14', 'Y')


# 5. Learn the graph
learnt_mixed_graph = learner.learnPDAG()



# 6. Visualize
print(" ")
print("--- CAUSAL DISCOVERY RESULTS ---")

# 1. Get Directed Arcs (->)
for tail, head in learnt_mixed_graph.arcs():
    mi = learner.correctedMutualInformation(tail, head)
    print(f"ARC: {dataset.columns[tail]} --> {dataset.columns[head]} | MI = {round(mi, 5)}")

# 2. Get Undirected Edges (-)
for u, v in learnt_mixed_graph.edges():
    print(f"UNDIRECTED EDGE: {dataset.columns[u]} --- {dataset.columns[v]}")
    

# 3. Get Bidirected Arcs (<->) 
# These represent Latent Common Causes identified by MIIC
latents = learner.latentVariables()
if latents:
    print(f"\nDetected {len(latents)} Latent Variable(s):")
    for lat in latents:
        # For gum.Arc, use .tail and .head (attributes, not methods)
        u, v = lat[0], lat[1]
        print(f"BIDIRECTIONAL (Latent): {dataset.columns[u]} <--> {dataset.columns[v]}")
else:
    print("\nNo Bidirected Arcs (Latents) detected.")
    
    
