import numpy as np

class IndependentLinearExplainer():
    def __init__(self, model, background):
        self.model = model
        self.background = background
        self.mean_feature_vals = background.mean()
        self.features = background.columns
        self.coefs = model.coef_
        self.intercept = model.intercept_

    @property
    def expected_value(self):
        return self.intercept + np.sum((self.mean_feature_vals * self.coefs))

    def explain(self, x):
        shap_values = []

        for _, row in x.iterrows():
            shap_sample = [
                coef * (row[feature] - self.mean_feature_vals[feature]) 
                for coef, feature in zip(self.coefs, self.features)
            ]
            shap_values.append(shap_sample)
        
        return shap_values