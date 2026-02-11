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
        learner.addForbiddenArc('Y', 'V_2') # Owning an EV "per se" does not change the number of cars owned. 
        learner.addForbiddenArc('Y', 'V_13') # Owning an EV does not cause tenancy
        learner.addForbiddenArc('V_2', 'V_7') # No. of cars does not cause parking provision
        learner.addForbiddenArc('V_8', 'V_9') # Dwelling type does not cause its age
        learner.addForbiddenArc('V_7', 'V_9') # Parking provision does not cause dwelling age
        learner.addForbiddenArc('V_7', 'V_13') # parking provision does not cause tenancy
        learner.addForbiddenArc('V_8', 'V_14') # dwelling type does not cause location (LA)
        learner.addForbiddenArc('V_14', 'V_8') # Location (LA) does not cause dwelling type
        learner.addForbiddenArc('V_2', 'V_8') # No. of cars does not cause dwelling type
        learner.addForbiddenArc('V_7', 'V_8') # parking provision does not cause dwelling type
        learner.addForbiddenArc('V_7', 'V_14') # parking provision does not cause location (LA)
        learner.addForbiddenArc('V_1', 'V_12') # income does not cause working status
        learner.addForbiddenArc('V_8', 'V_5') # dwelling type does not cause household composition
        learner.addForbiddenArc('V_1', 'V_5') # income does not change composition
        learner.addForbiddenArc('V_12', 'V_5') # working status does not change household composition
        learner.addForbiddenArc('V_9', 'V_6') # dwelling age does not cause urbanisation level
        learner.addForbiddenArc('V_9', 'V_12') # dwelling age does not cause household working status
        learner.addForbiddenArc('V_2', 'V_13') # No. cars does not cause tenancy
        learner.addForbiddenArc('V_13', 'V_2') # tenancy does not cause No. of cars owned
        learner.addForbiddenArc('V_7', 'V_13') # parking provision does not cause tenancy
        learner.addForbiddenArc('V_13', 'V_7') # tenancy does not cause parking provision
        learner.addForbiddenArc('V_12', 'V_9') # working status does not cause dwelling age
        learner.addForbiddenArc('V_6', 'V_9') # Urbanisation level does not cause dwelling age
        learner.addForbiddenArc('V_8', 'V_2') # dwelling type does not cause No. of cars
        
        # Mandatory edges
        learner.addMandatoryArc('V_7', 'Y') # parking provision causes EV ownership
        learner.addMandatoryArc('V_1', 'Y') # income causes EV ownership
        learner.addMandatoryArc('V_1', 'V_8') # income causes dwelling type
        learner.addMandatoryArc('V_1', 'V_13') # income causes tenancy
        learner.addMandatoryArc('V_7', 'V_2') # income causes No. of cars owned
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
    
    
