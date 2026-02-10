import pyAgrum as gum
import pandas as pd
from pathlib import Path
from pandas import DataFrame
from pyAgrum import PDAG, BNLearner



class CausalDiscovery():
    
    learned_MAG: PDAG = None
    dataset: DataFrame = None
    learner: BNLearner = None
    
    def discover_MAG(self) -> None:
 
        base_path = Path(r'C:/Causal_barriers-EV_uptake_local_code/Causal_barriers-EV_uptake/DATA')
        csv_path = base_path / 'processed_dataset.csv'
        
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
        learner.addForbiddenArc('V_1', 'V_12')
        learner.addForbiddenArc('V_8', 'V_5')
        learner.addForbiddenArc('V_1', 'V_5')
        learner.addForbiddenArc('V_12', 'V_5')
        learner.addForbiddenArc('V_9', 'V_6')
        learner.addForbiddenArc('V_9', 'V_12')
        learner.addForbiddenArc('V_2', 'V_13')
        learner.addForbiddenArc('V_13', 'V_2')
        learner.addForbiddenArc('V_7', 'V_13')
        learner.addForbiddenArc('V_13', 'V_7')

        # Mandatory edges
        learner.addMandatoryArc('V_7', 'Y')
        learner.addMandatoryArc('V_1', 'Y')
        learner.addMandatoryArc('V_1', 'V_8')
        learner.addMandatoryArc('V_1', 'V_13')
        learner.addMandatoryArc('V_10', 'Y')
        learner.addMandatoryArc('V_11', 'Y')
        #-------------------------------------

        self.learned_MAG = learner.learnPDAG()
        self.dataset = dataset
        self.learner = learner
        
        return


    def print_discovered_MAG(self) -> None:

        print(" ")
        print("--- CAUSAL DISCOVERY RESULTS (MAG) ---")

        # Directed Edges (called Arcs) (->)
        for tail, head in self.learned_MAG.arcs():
            print(f"Directed edge (Arc): {self.dataset.columns[tail]} --> {self.dataset.columns[head]}")

        # Undirected Edges (-)
        for u, v in self.learned_MAG.edges():
            print(f"Undirected edge: {self.dataset.columns[u]} --- {self.dataset.columns[v]}")
            
        # Bidirected Arcs (<->) representing Latent Common Causes identified via MIIC
        latents = self.learner.latentVariables()
        if latents:
            print(f"\nDetected {len(latents)} Latent Variable(s):")
            for lat in latents:
                u, v = lat[0], lat[1]
                print(f"BIDIRECTIONAL (Latent): {self.dataset.columns[u]} <--> {self.dataset.columns[v]}")
        else:
            print("\nNo Bidirected Arcs (Latents) detected.")
            
        return
    
    
