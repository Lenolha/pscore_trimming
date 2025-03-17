import pandas as pd
import numpy as np

# Funktion zum Discarding der Propensity Scores, die unterhalb oder oberhalb eines bestimmten Schwellenwerts liegen
def pscore_discard(data, pscore_est, smpls, trimming_value):
    removed_idx = np.where(((pscore_est < trimming_value) | (pscore_est > (1 - trimming_value))))[0]
    data_trimmed = data.drop(index=removed_idx)
    n_obs = data.shape[0]
    folds_list = np.empty(n_obs, dtype=int)
    for i in range(5):
        folds_list[smpls[0][i][1]] = i
    folds_list = np.delete(folds_list, removed_idx)
    smpls_new = []
    for i in range(5):
        smpls_new.append((np.where(folds_list != i)[0],np.where(folds_list == i)[0]))
    smpls_new = [smpls_new]
    pscore_trimmed = np.delete(pscore_est, removed_idx).reshape(-1,1)
    return smpls_new, data_trimmed, pscore_trimmed