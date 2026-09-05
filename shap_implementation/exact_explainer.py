import math
import numpy as np
from itertools import combinations

class ExactExplainer():

    def __init__(self, model, background):
        self.model = model
        self.background = background

    @property
    def expected_value(self):
        return self.coalition_value(None, set())

    def coalition_value(self, x, S):
        background = self.background.copy()

        for feature in S:
            background.iloc[:, feature] = x.iloc[feature]
        
        return self.model.predict(background).mean()


    def all_subsets(self, features):
        subsets = []

        for r in range(len(features)+1):
            subsets.extend(combinations(features, r))
        
        return subsets


    def shap_value(self, x, i):
        n = self.background.shape[1]
        phi = 0
        subsets = self.all_subsets(range(n))

        for S in subsets:
            S = set(S)

            if i in S:
                continue

            v_without_i = self.coalition_value(x, S)
            v_with_i = self.coalition_value(x, S | {i})
            weight = (
                math.factorial(len(S))
                * math.factorial(n - len(S) - 1)
                / math.factorial(n)
            )
            phi += weight * (v_with_i - v_without_i)
        
        return phi
    
    def explain(self, x):
        return np.array([
            [
                self.shap_value(x_row, i)
                for i in range(x.shape[1])
            ]
            for _, x_row in x.iterrows()
        ])




class ExactExplainerOptimized():
    
    def __init__(self, model, background):
        self.model = model
        self.background = background
        self.cache = {}
        self.n = background.shape[1]
        self.weights = self.precomp_weights()
        self.subsets = self.all_subsets()
    
    @property
    def expected_value(self):
        return self.coalition_value(None, frozenset())


    def coalition_value(self, x, S):
        if S in self.cache:
            return self.cache[S]

        background = self.background.copy()

        for feature in S:
            background.iloc[:, feature] = x.iloc[feature]
        
        mean_pred = self.model.predict(background).mean()
        self.cache[S] = mean_pred
        
        return mean_pred

    def all_subsets(self):
        return [
            frozenset(S) 
            for r in range(self.n + 1)
            for S in combinations(range(self.n), r)
        ]
    
    def precomp_weights(self):
        n_factorial = math.factorial(self.n)
        return {r: 
            math.factorial(r)
            * math.factorial(self.n - r - 1)
            / n_factorial
            for r in range(self.n)
        }

    def shap_value(self, x, i):
        phi = 0

        for S in self.subsets:
            if i in S:
                continue
            
            v_without_i = self.coalition_value(x, S)
            v_with_i = self.coalition_value(x, S | {i})
            phi += self.weights[len(S)] * (v_with_i - v_without_i)
        
        return phi
    
    def explain(self, x):
        shap_values = []

        for _, x_row in x.iterrows():
            self.cache.clear()

            row_shap_values = [self.shap_value(x_row, i) for i in range(self.n)]

            shap_values.append(row_shap_values)

        return np.array(shap_values)

