import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    initial_transition,
    initial_phi,
    initial_epsilon
)

class KCNode:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parents = []
        self.question_count = 0
        
        # Posterior
        self.downward_alpha = [0, 0]
        self.upward_beta = [0, 0]
        self.posterior1 = None
        self.posterior2 = None
        self.posterior3 = None


    def add_child(self, child_node):
        self.children.append(child_node)
    
    def add_parent(self, parent_node):
        self.parents.append(parent_node)
    

    def to_dict(self):
        """Convert node to dictionary for saving"""
        return {
            "name": self.name,
            "question_count": self.question_count,

            # Posterior
            "downward_alpha": self.downward_alpha,
            "upward_beta": self.upward_beta,
            "posterior1": self.posterior1,
            "posterior2": self.posterior2,
            "posterior3": self.posterior3,

            # Children and parents  
            "children": [child.name for child in self.children],  # Store children as names
            "parents": [parent.name for parent in self.parents],  # Store parents as names
        }
    
class ParametersNode:
    def __init__(self, name):
        self.name = name

        # Parameters
        self.gamma = initial_transition # Transition probability
        self.gamma_root = None # Prior probability of root node
        self.phi = initial_phi # Emission probability when K_M(t)i = 1
        self.epsilon = initial_epsilon # Emission probability when K_M(t)i = 0
        self.r_diff = None # Calibration parameter
        
    def to_dict(self):
        """Convert node to dictionary for saving"""
        return {
            "name": self.name,
            "gamma": self.gamma,
            "gamma_root": self.gamma_root,
            "phi": self.phi,
            "epsilon": self.epsilon,
            "r_diff": self.r_diff,
        }

