





import pyagrum as gum

# 1. Initialize the learner with your dataframe
learner = gum.BNLearner(self.ds_obsrv_vars)


# 3. Choose your discovery algorithm
# MIIC is excellent for causal discovery in pyAgrum
learner.useMIIC() 

# 4. Optional: Add structural constraints (Domain Knowledge)
# We know off-street parking cannot be CAUSED by EV ownership (temporality)
learner.addForbiddenArc("ev_ownership", "parking_provision")

# 5. Learn the graph
learnt_dag = learner.learnDAG()

# 6. Visualize
import pyagrum.lib.notebook as gnb
gnb.showBN(learnt_dag)