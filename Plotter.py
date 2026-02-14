from CausalGraphicalModel import CausalGraphicalModel
import pyAgrum.causal as csl
from pyAgrum import Potential
import pyAgrum as gum
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


class Plotter():
    causal_grap_model: CausalGraphicalModel = None
    P_Y_do_V_7_potential: Potential = None
    P_Y_given_V_7_potential: Potential = None
    P_Y_do_V_1_potential: DataFrame = None
    
    def __init__(self):
        self.causal_grap_model = CausalGraphicalModel(dataset_name='processed_dataset.csv')
        self.causal_grap_model.build()
        
        self._get_P_Y_do_V_7_potential() # interventional distribution P(EV | do(Parking provision))
        self._get_P_Y_given_V_7_potential() # observational distribution P(EV | Parking provision)
        self._get_P_Y_given_V_1_potential() # 2-fold interventional distribution P(EV | do(Income))
        return
    
    
    def _get_P_Y_do_V_7_potential(self):
        cm = self.causal_grap_model.c_model # the pyAgrum CausalModel object
        formula, potential, explanation = csl.causalImpact(cm, on='Y', doing='V_7', knowing={})
        self.P_Y_do_V_7_potential = potential
        
        return
    
    
    def _get_P_Y_given_V_7_potential(self):
        ve = gum.VariableElimination(self.causal_grap_model.b_net)
        self.P_Y_given_V_7_potential = ve.evidenceJointImpact(targets=['Y'], evs={'V_7'}) #returns a pyAgrum.Potential for P(targets|evs) for all instantiations (values) of targets and evs variables.

        return
    
    
    def _get_P_Y_given_V_1_potential(self):
        '''
        NOTE: it returns the causal potential with income var (V_1) as
        a two-fold {low-income; high-income} instead of 8-fold category, 
        split at the median income value (approx. £30k)
        '''
        cm = self.causal_grap_model.c_model # the pyAgrum CausalModel object
        formula, potential, explanation = csl.causalImpact(cm, on='Y', doing='V_1', knowing={})
        ve = gum.VariableElimination(self.causal_grap_model.b_net)
        p_V_1 = ve.evidenceJointImpact(targets=['V_1'], evs={}) 

        arr_Prob_Y_do_V1 = potential.toarray()
        arr_Prob_V_1 = p_V_1.toarray()

        # Slice for Low Income (Bands 1-6 -> indices 0,1,2,3,4,5)
        low_pot = arr_Prob_Y_do_V1[0:6, :]
        low_weights = arr_Prob_V_1[0:6]

        # Slice for High Income (Bands 7-8 -> indices 6,7)
        high_pot = arr_Prob_Y_do_V1[6:8, :]
        high_weights = arr_Prob_V_1[6:8]

        low_dist = self._weighted_avg(pot=low_pot, weights=low_weights)
        high_dist = self._weighted_avg(pot=high_pot, weights=high_weights)

        two_fold_potential = pd.DataFrame(
            [low_dist, high_dist], 
            index=['Low Income (V_1 vals: 1 to 6)', 'High Income (V_1 vals: 7 to 8)'],
            columns=['Y val=1', 'Y val=2', 'Y val=3', 'Y val=4']
        )

        #print("2-fold Post-treatment distribution P(Y | do(V_1)):")
        #print("______________________________________________________")
        #print(two_fold_potential)
        
        
        self.P_Y_do_V_1_potential = two_fold_potential
        
        return
    
    
    def _weighted_avg(self, pot, weights):
        numerator = np.sum(pot * weights[:, np.newaxis], axis=0)
        denominator = np.sum(weights)
        return numerator / denominator
    
    
    
    def generate_income_plots(self, fig_name: str):
        df_interventional = self.P_Y_do_V_1_potential
        
        x_labels = ["Already own\nelectric car/van", "Thinking to\nbuy one soon", "Thinking to buy\none in the future", "Not considering\nto buy one"]
        plt.rcParams["font.family"] = "Arial"
        fig, axes = plt.subplots(2, 1, figsize=(5., 9), gridspec_kw={'height_ratios': [1.7, 1]})  # 2 rows x 1 col
        
        #---------------------------------------------------------------------------
        ax = axes[0]
        
        int_values_LOW_income = df_interventional.values.tolist()[0]
        int_values_HIGH_income = df_interventional.values.tolist()[1]
        
        bar_width = 0.4
        b_1 = np.arange(len(x_labels))
        b_2 = [i + bar_width for i in b_1]
                        
        ax.bar(b_1, int_values_LOW_income, bar_width, label="Control: " r"$V_1 = \text{low-income}$", color="#618BCF", edgecolor='black')
        ax.bar(b_2, int_values_HIGH_income, bar_width, label="Treatment: " r"$V_1 = \text{high-income}$", color="#2259B2", edgecolor='black') 
        
        ax.set_xlabel('EV ownership status & intention (Y)', fontsize=14)
        ax.set_ylabel("Post-intervention probability:  " r"$P(Y \mid \text{do}(V_1))$", fontsize=14)
        ax.set_xticks(b_1 + (bar_width - 0.2) / 2, x_labels, rotation=90)
        plt.setp(ax.get_xticklabels(), ha='left', va='top') 
        ax.set_ylim(0, 0.7)
        
        ax.tick_params(axis='x', labelsize=12, length=0)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7, color='black')
        ax.legend(loc='upper left', frameon=True, title="Income intervention", title_fontsize=12, fontsize=12, framealpha=1.0)
        
        ax.text(-0.1, 1.15, "a", transform=ax.transAxes, fontsize=19, fontweight='bold', va='top')
        #--------------------------------------------------------------------------------------------
        ax = axes[1]
        
        treatment_effects = [(off_strt - on_strt)*100. for off_strt, on_strt in zip(int_values_HIGH_income, int_values_LOW_income)]
        
        bar_width = 0.8
        b_income_interv = np.arange(len(x_labels))
        
        ax.bar(b_income_interv, treatment_effects, bar_width, color="#49C5A2", edgecolor='black')
        
        ax.set_xticks(b_income_interv, ['', '', '', ''])
        ax.set_ylabel("Treatment effect: " r"$\Delta_{TE}$" " (pp)", fontsize=14)
        
        ax.set_ylim(-30, 20)
        
        ax.tick_params(axis='x', labelsize=12, length=5)
        ax.tick_params(axis='y', labelsize=12)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.xaxis.tick_top()
        
        ax.text(b_income_interv[0], treatment_effects[0] + 0.1, f"+{round(treatment_effects[0], 1)}", ha='center', va='bottom', fontsize=12)
        ax.text(b_income_interv[1], treatment_effects[1] + 0.1, f"+{round(treatment_effects[1], 1)}", ha='center', va='bottom', fontsize=12)
        ax.text(b_income_interv[2], treatment_effects[2] + 0.1, f"+{round(treatment_effects[2], 1)}", ha='center', va='bottom', fontsize=12)
        ax.text(b_income_interv[3], treatment_effects[3] - 1, f"-{round(treatment_effects[3], 1)}", ha='center', va='top', fontsize=12)
        
        ax.text(-0.1, 1.2, "b", transform=ax.transAxes, fontsize=19, fontweight='bold', va='top')
        #----------------------------------------------------------------------------------
        
        plt.subplots_adjust(hspace=10)
        plt.tight_layout(h_pad=3.0)
        plt.subplots_adjust(top=0.935)
        plt.savefig(f"FIGURES/{fig_name}.jpeg")
        return
    
    
    
    def generate_parking_provision_plots(self, fig_name: str):
        df_interventional = self.P_Y_do_V_7_potential.topandas()
        df_observational = self.P_Y_given_V_7_potential.topandas()
        
        x_labels = ["Already own\nelectric car/van", "Thinking to\nbuy one soon", "Thinking to buy\none in the future", "Not considering\nto buy one"]
        
        plt.rcParams["font.family"] = "Arial"
        fig, axes = plt.subplots(2, 2, figsize=(9, 9), gridspec_kw={'height_ratios': [1.7, 1]})  # 2 rows x 2 cols
        

        #---------------------------------------------------------------------------
        ax = axes[0, 0]
        
        int_values_OFF_street = df_interventional.values.tolist()[0]
        int_values_ON_street = df_interventional.values.tolist()[1]
        
        bar_width = 0.4
        b_1 = np.arange(len(x_labels))
        b_2 = [i + bar_width for i in b_1]
                        
        ax.bar(b_1, int_values_ON_street, bar_width, label="Control: " r"$V_7 = \text{on-street}$", color="#618BCF", edgecolor='black')
        ax.bar(b_2, int_values_OFF_street, bar_width, label="Treatment: " r"$V_7 = \text{off-street}$", color="#2259B2", edgecolor='black')
        
        ax.set_xlabel('EV ownership status & intention (Y)', fontsize=14)
        ax.set_ylabel("Post-intervention probability:  " r"$P(Y \mid \text{do}(V_7))$", fontsize=14)
        ax.set_xticks(b_1 + (bar_width - 0.2) / 2, x_labels, rotation=90)
        plt.setp(ax.get_xticklabels(), ha='left', va='top') 
        ax.set_ylim(0, 0.6)
        
        ax.tick_params(axis='x', labelsize=12, length=0)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7, color='black')
        ax.legend(loc='upper left', frameon=True, title="Parking provision intervention", title_fontsize=12, fontsize=12, framealpha=1.0)
        
        ax.text(-0.06, 1.1, "a", transform=ax.transAxes, fontsize=19, fontweight='bold', va='top')
        #------------------------------------------------------------------------------


        ax = axes[0, 1]
        
        obs_values_OFF_street = df_observational.values.tolist()[0]
        obs_values_ON_street = df_observational.values.tolist()[1]
        
        bar_width = 0.4
        b_1 = np.arange(len(x_labels))
        b_2 = [i + bar_width for i in b_1]
                        
        ax.bar(b_1, obs_values_ON_street, bar_width, label="Obsv: " r"$V_7 = \text{on-street}$", color="#FFA14F", edgecolor='black')
        ax.bar(b_2, obs_values_OFF_street, bar_width, label="Obsv: " r"$V_7 = \text{off-street}$", color="#D67018", edgecolor='black')
        
        ax.set_xlabel('EV ownership status & intention (Y)', fontsize=14)
        ax.set_ylabel("Observational probability:  " r"$P(Y \mid V_7)$", fontsize=14)
        ax.set_xticks(b_1 + (bar_width - 0.2) / 2, x_labels, rotation=90)
        plt.setp(ax.get_xticklabels(), ha='left', va='top') 
        ax.set_ylim(0, 0.6)
        
        ax.tick_params(axis='x', labelsize=12, length=0)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7, color='black')
        ax.legend(loc='upper left', frameon=True, title="Parking provision observation", title_fontsize=12, fontsize=12, framealpha=1)
        
        ax.text(-0.06, 1.1, "b", transform=ax.transAxes, fontsize=19, fontweight='bold', va='top')
        #------------------------------------------------------------------------------
        
        ax = axes[1, 0]
        
        
        treatment_effects = [(off_strt - on_strt)*100. for off_strt, on_strt in zip(int_values_OFF_street, int_values_ON_street)]
        
        bar_width = 0.8
        b_pp_interv = np.arange(len(x_labels))
        
        ax.bar(b_pp_interv, treatment_effects, bar_width, color="#49C5A2", edgecolor='black')
        
        ax.set_xticks(b_pp_interv, ['', '', '', ''])
        ax.set_ylabel("Treatment effect: " r"$\Delta_{TE}$" " (pp)", fontsize=14)
        
        ax.set_ylim(-6.5, 4.5)
        
        ax.tick_params(axis='x', labelsize=12, length=5)
        ax.tick_params(axis='y', labelsize=12)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.xaxis.tick_top()
        
        ax.text(b_pp_interv[0], treatment_effects[0] + 0.1, f"+{round(treatment_effects[0], 1)}", ha='center', va='bottom', fontsize=12)
        ax.text(b_pp_interv[1], treatment_effects[1] + 0.1, f"+{round(treatment_effects[1], 1)}", ha='center', va='bottom', fontsize=12)
        ax.text(b_pp_interv[2], treatment_effects[2] - 0.15, f"-{round(treatment_effects[2], 1)}", ha='center', va='top', fontsize=12)
        ax.text(b_pp_interv[3], treatment_effects[3] + 0.1, f"+{round(treatment_effects[3], 1)}", ha='center', va='bottom', fontsize=12)
        
        ax.text(-0.06, 1.15, "c", transform=ax.transAxes, fontsize=19, fontweight='bold', va='top')
        #----------------------------------------------------------------------------------
        
        ax = axes[1, 1]
        
        
        observed_diff = [(off_strt - on_strt)*100. for off_strt, on_strt in zip(obs_values_OFF_street, obs_values_ON_street)]
        
        bar_width = 0.8
        b_pp_obs = np.arange(len(x_labels))
        
        ax.bar(b_pp_obs, observed_diff, bar_width, color="#49C5A2", edgecolor='black')
        
        ax.set_xticks(b_pp_obs, ['', '', '', ''])
        ax.set_ylabel("Observed difference: " r"$\Delta_{Obsv.}$" " (pp)", fontsize=14)
        
        ax.set_ylim(-6.5, 4.5)
        
        ax.tick_params(axis='x', labelsize=12, length=5)
        ax.tick_params(axis='y', labelsize=12)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.xaxis.tick_top()
        
        ax.text(b_pp_obs[0], observed_diff[0] + 0.1, f"+{round(observed_diff[0], 1)}", ha='center', va='bottom', fontsize=12)
        ax.text(b_pp_obs[1], observed_diff[1] + 0.1, f"+{round(observed_diff[1], 1)}", ha='center', va='bottom', fontsize=12)
        ax.text(b_pp_obs[2], observed_diff[2] - 0.15, f"-{round(observed_diff[2], 1)}", ha='center', va='top', fontsize=12)
        ax.text(b_pp_obs[3], observed_diff[3] - 0.15, f"-{round(observed_diff[3], 1)}", ha='center', va='top', fontsize=12)
        
        ax.text(-0.06, 1.15, "d", transform=ax.transAxes, fontsize=19, fontweight='bold', va='top')
        #----------------------------------------------------------------------------------
        

            
        
        
        plt.subplots_adjust(hspace=10)
        plt.tight_layout(h_pad=3.0)
        plt.subplots_adjust(top=0.935)
        plt.savefig(f"FIGURES/{fig_name}.jpeg")
        
        return
    
    