import pyAgrum as gum
import pyAgrum.lib.image as gimg
import pandas as pd
import os
os.environ["PATH"] += r";C:\Program Files\Graphviz\bin"



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
# because the caputed correlation is to be expained in 
# terms of a confounder (latent or observed).
learner.addForbiddenArc('Y', 'V_7') 
learner.addForbiddenArc('Y', 'V_7') 

# You would force a mandatory arc because of domain 
# knowledge (such as a known physical law or a temporal 
# sequence) that a causal relationship exists.
learner.addMandatoryArc('V_7', 'Y')

# 5. Learn the graph
learnt_mixed_graph = learner.learnPDAG() #learnEssentialGraph() #



# 6. Visualize
print("--- CAUSAL DISCOVERY RESULTS ---")

# 1. Get Directed Arcs (->)
for tail, head in learnt_mixed_graph.arcs():
    print(f"ARC: {dataset.columns[tail]} --> {dataset.columns[head]}")

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
    
    
