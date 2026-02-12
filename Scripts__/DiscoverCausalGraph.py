from CausalDiscovery import CausalDiscovery


'''
Script to discover the partially directed graph (a Maximal Ancestral Graphs, MAG) from the data. 
It uses the Multivariate Information-based Inductive Causation (MIIC) alghoritm, developed by the Isambert Lab at the Curie institute.
NOTE: the outputtted MAG includes undirected edges, hence it requires manual refinement to run causal identification/estimation.
'''

 
   
cd = CausalDiscovery()
cd.discover_MAG(dataset_name='processed_dataset.csv')
cd.print_discovered_MAG()