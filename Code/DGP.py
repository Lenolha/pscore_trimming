import pandas as pd
import numpy as np
from scipy.linalg import toeplitz

# Funktionen zur Erzeugung von Propensity Scores 
def propensity_eq(x, cov_mat, R2_d):
    dim_x = x.shape[1]
    beta = [1 / (k**2) for k in range(1, dim_x + 1)]
    b_sigma_b = np.dot(np.dot(cov_mat, beta), beta)
    c_d = np.sqrt(np.pi**2 / 3. * R2_d/((1-R2_d) * b_sigma_b))
    xx = np.exp(np.dot(x, np.multiply(beta, c_d))) #Die Berechnung der Propensity Scores ist nichtlinear aufgrund der Verwendung der Exponentialfunktion.
    propensity_score = (xx/(1+xx))
    return propensity_score

# Funktionen zur Erzeugung von Potential Outcomes
def potential_outcome_eq(x, cov_mat, R2_y, theta):
    dim_x = x.shape[1]
    beta = [1 / (k**2) for k in range(1, dim_x + 1)]
    b_sigma_b = np.dot(np.dot(cov_mat, beta), beta)
    c_y = np.sqrt(R2_y/((1-R2_y) * b_sigma_b))
    zeta = np.random.standard_normal(size=[x.shape[0], ]) #Zufallszahl aus Standardnormalverteilung
    d = 0
    Y_0 = d * theta + d * np.dot(x, np.multiply(beta, c_y)) + zeta
    d = 1
    Y_1 = d * theta + d * np.dot(x, np.multiply(beta, c_y)) + zeta
    return Y_0, Y_1

# Funktion zur Erzeugung von Daten für das interaktive Regressionsmodell (IRM)
def make_irm_data(n_obs, dim_x, theta, R2_d, R2_y):
    """
    Variation of DoubleML's make_irm_data function. Generates data for the interactive regression model (IRM) example and returns potential values
    References
    ----------
    Belloni, A., Chernozhukov, V., Fernández‐Val, I. and Hansen, C. (2017). Program Evaluation and Causal Inference With
    High‐Dimensional Data. Econometrica, 85: 233-298.
    """
    # inspired by https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA12723, see suplement
    v = np.random.uniform(size=[n_obs, ]) #Zufallszahl zwischen 0 und 1
    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ]) #abnehmende Korrelationen zwischen den x-Variablen
    propensity_score = propensity_eq(x, cov_mat, R2_d)
    D = 1. * (propensity_score > v)
    Y_0, Y_1 = potential_outcome_eq(x, cov_mat, R2_y, theta)
    Y = (1 - D) * Y_0 + D * Y_1 
    x_flat = pd.DataFrame(x, columns=[f'x_{i}' for i in range(x.shape[1])])
    df_orcl = pd.concat([pd.DataFrame({'Y_0': Y_0, 'Y_1': Y_1, 'ps': propensity_score, 'D': D, 'Y': Y}), x_flat], axis=1)
    df = pd.concat([pd.DataFrame({'D': D, 'Y': Y}), x_flat], axis=1)
    results = {'df': df, 'df_orcl': df_orcl}
    return results


