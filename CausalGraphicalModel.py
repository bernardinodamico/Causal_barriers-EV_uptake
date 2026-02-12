import pyAgrum as gum
import pandas as pd
import pyAgrum.causal as csl
from pyAgrum.causal import CausalModel
import networkx as nx
from networkx import DiGraph
from pyAgrum import BayesNet
from Values_mapping import GetVariableValues as vvalues
from pandas import DataFrame
from CausalDiscovery import CausalDiscovery
from pathlib import Path


class CausalGraphicalModel():
    dataset_filename: str = None
    disctetised_ds: DataFrame = None 
    b_net: BayesNet = None
    c_model: CausalModel = None
    G: DiGraph = None
    Lp_smoothing: float = None
 
 
    def __init__(self, dataset_name: str):
        """
        parameter: dataset_filename = the name of the training dataset (including its file extention)
        """ 
        base_path = Path(r'C:/Causal_barriers-EV_uptake_local_code/Causal_barriers-EV_uptake/DATA')
        csv_path = base_path / dataset_name
        
        self.disctetised_ds = pd.read_csv(filepath_or_buffer=csv_path, sep=",")
        self.dataset_filename = dataset_name

        return
    
    def set_Lp_smoothing(self, Lp_sm: float):
        self.Lp_smoothing = Lp_sm
        return
    

    def add_nodes(self) -> None:
        self.b_net = gum.BayesNet("MyCausalBN")
        
        self.b_net.add(gum.LabelizedVariable('Y', "EV ownership status & intention" , vvalues.get_nums(var_symbol='Y'))) 

        self.b_net.add(gum.LabelizedVariable('V_1', "Household income" , vvalues.get_nums(var_symbol='V_1'))) 
        self.b_net.add(gum.LabelizedVariable('V_2', "No. of vehicles" , vvalues.get_nums(var_symbol='V_2'))) 
        self.b_net.add(gum.LabelizedVariable('V_3', "No. of adults in household" , vvalues.get_nums(var_symbol='V_3'))) 
        self.b_net.add(gum.LabelizedVariable('V_4', "No. of children in household" , vvalues.get_nums(var_symbol='V_4'))) 
        self.b_net.add(gum.LabelizedVariable('V_5', "Household composition" , vvalues.get_nums(var_symbol='V_5'))) 
        self.b_net.add(gum.LabelizedVariable('V_6', "Urbanisation level" , vvalues.get_nums(var_symbol='V_6')))
        self.b_net.add(gum.LabelizedVariable('V_7', "Parking provision" , vvalues.get_nums(var_symbol='V_7')))
        self.b_net.add(gum.LabelizedVariable('V_8', "Dwelling type" , vvalues.get_nums(var_symbol='V_8')))
        self.b_net.add(gum.LabelizedVariable('V_9', "Dwelling age" , vvalues.get_nums(var_symbol='V_9')))
        self.b_net.add(gum.LabelizedVariable('V_10', "Workplace charging infrastructure density" , vvalues.get_nums(var_symbol='V_10')))  
        self.b_net.add(gum.LabelizedVariable('V_11', "Public charging infrastructure density" , vvalues.get_nums(var_symbol='V_11')))
        self.b_net.add(gum.LabelizedVariable('V_12', "Household working status" , vvalues.get_nums(var_symbol='V_12')))
        self.b_net.add(gum.LabelizedVariable('V_13', "Tenancy" , vvalues.get_nums(var_symbol='V_13')))
        self.b_net.add(gum.LabelizedVariable('V_14', "Local authority" , vvalues.get_nums(var_symbol='V_14'))) 

        return
    

    def add_causal_edges(self) -> None:
        
        # run the causal discovery (MIIC) algo
        cd = CausalDiscovery()
        cd.discover_MAG(dataset_name=self.dataset_filename)
        
        # add directed edges as found by the MIIC algo
        for tail, head in cd.learned_MAG.arcs():
            self.b_net.addArc(cd.dataset.columns[tail], cd.dataset.columns[head]) 
        
        # manually add edges that MIIC was unable to direct
        #self.b_net.addArc('V_14', 'V_11') 


        return
    

    def learn_params(self, data: DataFrame) -> None:
        learner = gum.BNLearner(data, self.b_net)
        learner.useSmoothingPrior(self.Lp_smoothing) # Laplace smoothing (e.g. a count C is replaced by C+1)
        self.b_net = learner.learnParameters(self.b_net.dag())
        return
    

    def add_latent_vars(self)-> None:
        '''
        Method to add latent variables to the PyAgrum CausalModel object "c_model". 
        '''
        self.c_model = csl.CausalModel(bn=self.b_net, 
                                  latentVarsDescriptor=[("U_1", ["V_1","V_2"]),
                                                        ("U_2", ["V_8","Y"]),
                                                        ],
                                  keepArcs=True)
        return
    

    def build(self) -> None:
        self.set_Lp_smoothing(Lp_sm=0.005)
        self.add_nodes()
        self.add_causal_edges()
        self.learn_params(data=self.disctetised_ds)
        self.add_latent_vars()
        return
    
    





