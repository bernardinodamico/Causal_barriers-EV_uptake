import pandas as pd
from pandas import DataFrame
import os
import math
import numpy as np
#from Values_mapping import GetVariableValues
import time
import datetime


from Values_mapping import *



def _compute_infrastruct_densities() -> None:
        
    Workplace_charging_ds = pd.read_excel(io=os.path.join(os.path.dirname(__file__), r"DATA\RAW\electric-vehicle-charging-device-grant-scheme-statistics-january-2023.xlsx"), sheet_name=f"3", header=2)
    OnStreetRes_charging_ds = pd.read_excel(io=os.path.join(os.path.dirname(__file__), r"DATA\RAW\electric-vehicle-charging-device-grant-scheme-statistics-january-2023.xlsx"), sheet_name=f"combined", header=1)
    Public_charging_ds = pd.read_excel(io=os.path.join(os.path.dirname(__file__), r"DATA\RAW\electric-vehicle-public-charging-infrastructure-statistics-october-2025.xlsx"), sheet_name=f"1a", header=2)
    pop_estimate = pd.read_excel(io=os.path.join(os.path.dirname(__file__), r"DATA\RAW\data-mid-year-population-estimates-2022.xlsx"), sheet_name=f"Table 1", header=3)

    council_pops = []
    for i in range(3, 35):
        council_pops.append(pop_estimate.loc[i, "All ages"])

    council_labels = []
    workplace_installs = []
    for i in range(354, 386):
        council_labels.append(Workplace_charging_ds.loc[i, "Region or Local Authority"])
        workplace_installs.append(Workplace_charging_ds.loc[i, "Sockets Installed"])
    
    onStreetRes_installs = []
    for i in range(0, 32):
        onStreetRes_installs.append(OnStreetRes_charging_ds.loc[i, "Total"])
        
    public_installs = []
    for i in range(389, 421):
        public_installs.append(Public_charging_ds.loc[i, "Jan-23"])
        
    
    vv = VariableValues()
    V_10_vlabels = list(vv.Variables_dic['V_10'].values())
    V_10_vnums = list(vv.Variables_dic['V_10'].keys())
    
    #----------------------------------------------------------------------------------------
    wp_installs = {}
    os_installs = {}
    pb_installs = {}
    
    for i in range(0, len(V_10_vlabels)):
        for j in range(0, len(council_labels)):
            if V_10_vlabels[i] == council_labels[j]:
                per_capita_workplace_installs = (workplace_installs[j] / council_pops[j]) * 100000
                wp_installs[V_10_vnums[i]] = per_capita_workplace_installs
                
                per_capita_OnStreetRes_installs = (onStreetRes_installs[j] / council_pops[j]) * 100000
                os_installs[V_10_vnums[i]] = per_capita_OnStreetRes_installs
                
                per_capita_public_install = (public_installs[j] / council_pops[j]) * 100000
                pb_installs[V_10_vnums[i]] = per_capita_public_install
                
                break
            else:
                continue
    
    return wp_installs, os_installs, pb_installs


'''
NOTE: the returend densities (per 1000000 persons) need be discretised. Make the above an internal method and
call it inside the "fill_in_infrastruct_density()", where the dictionaries wp_installs, os_installs, pb_installs 
are used as look up tables to enter the value of each row in the processed_dataset as a function of the "council" variable
'''

wp_installs, os_installs, pb_installs = _compute_infrastruct_densities()