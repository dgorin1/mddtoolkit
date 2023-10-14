from cyipopt import Problem
from jax import numpy as jnp
import jax

class DiffusionProblem(Problem):
    def __init__(self, objective, bounds):
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]
        
        constraint_lb = [0,0]
        constraint_ub = [jnp.inf,0]
        self.n_constraints = 2
        
        super().__init__(n=len(bounds), m=2, lb=lb, ub=ub, cl=constraint_lb, cu=constraint_ub)
        
        self._objective = objective
        self.constraint_jacobian = jax.jacfwd(self.constraints)
        
    
    def objective(self, X):
        return self._objective(X)
    
    
    def gradient(self, X):
        return self._objective.grad(X)
    
    
    def constraints(self, X):

        X = X[1:]
        if len(X) <= 3:
            ndom = 1
        else:
            ndom = (len(X))//2
        
        # sum of fracs must be at most 1
        temp = X[1:]
        fracstemp = temp[ndom:]

        frac_constraint = 1-jnp.sum(fracstemp)
        
        # lnd0aa must be in decreasing order.
        lnD0aa = temp[0:ndom]

        lnD0aa_constraint = 0
        for i in range(len(lnD0aa)-1):
            if lnD0aa[i]-lnD0aa[i+1] <= 0:
                lnD0aa_constraint += max(lnD0aa[i]-lnD0aa[i+1], 0)
                
        return jnp.array([frac_constraint, lnD0aa_constraint])
    
    
    def jacobian(self, X):
        return self.constraint_jacobian(X)
    

    # def intermediate(
    #         self,
    #         alg_mod,
    #         iter_count,
    #         obj_value,
    #         inf_pr,
    #         inf_du,
    #         mu,
    #         d_norm,
    #         regularization_size,
    #         alpha_du,
    #         alpha_pr,
    #         ls_trials
    #         ):

    #     global min_fx
    #     global xBest
    #     print(min_fx)
    #     if obj_value < min_fx:
    #         min_fx = obj_value
    #         xBest = X

