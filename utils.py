
import pandas as pd, numpy as np
from scipy.stats import norm, bernoulli, uniform


def fuzzy_prob(centered_x:list|np.ndarray, 
               std=0.1, # Uncertainty around the cutoff
               ) -> np.ndarray: # Expected value of treatment at the cutoff
    """Define treatment probability for fuzzy design"""
    
    centered_x = np.array(centered_x)
    #const = norm.ppf(cutoff_prob) # constant to adjust the probability at the cutoff
    #prob = norm.cdf(centered_x / std + const)
    prob = norm.cdf(centered_x / std)
    return prob


def fuzzy_treatment_assignment(centered_x:list|np.ndarray, 
                               std=0.1, # Uncertainty around the cutoff
                               seed=None) -> np.ndarray: # seed for reproducibility
    """
    Assigns treatment based on a fuzzy design around a cutoff.
    The treatment is always assigned to the right of the cutoff.
    """

    prob = fuzzy_prob(centered_x, std=std)
    rng = np.random.default_rng(seed=int(seed)) # set seed
    treat_assigned = bernoulli.rvs(p=prob, random_state=rng)
    #treat_assigned = pd.Series(treat_assigned, index=centered_x.index)
    
    return treat_assigned


def sharp_treatment_assignment(centered_x: list|np.ndarray) -> np.ndarray: # Cutoff value for treatment assignment

    """
    Assigns treatment based on a sharp design around a cutoff.
    The treatment is always assigned to the right of the cutoff.
    """    
    centered_x = np.array(centered_x)
    treat_assigned = (centered_x >= 0).astype(int)
    
    return treat_assigned
