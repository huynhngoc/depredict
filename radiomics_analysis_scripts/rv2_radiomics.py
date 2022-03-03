import hoggorm as ho
import pandas as pd
import numpy as np


labels = [10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58]
bins = [4, 8, 16, 32, 48, 64]
periods = ['bl', 'pt']

filename = 'radiomics_outputs/{period}_bin{bin:02d}_label{label:02d}.csv'
name_list = []
data_list_centered = []
data_list_standardized = []
for period in periods:
    for label in labels:
        for bin in bins:
            print(period, label, bin)
            data = pd.read_csv(filename.format(period=period, label=label, bin=bin), index_col=0)
            data_list_centered.append(ho.center(data.values, axis=0))
            data_list_standardized.append(ho.standardise(data.values, mode=0))
            name_list.append(f'{period}_bin{bin:02d}_label{label:02d}')

print('Calculating rv2 centered')
rv2_results_centered = ho.RV2coeff(data_list_centered)
print('Calculating rv2 standardized')
rv2_results_standardized = ho.RV2coeff(data_list_standardized)

print('saving to files')
pd.DataFrame(rv2_results_centered, index=name_list, columns=name_list).to_csv('radiomics_outputs_analysis/rv2_centered.csv')
pd.DataFrame(rv2_results_standardized, index=name_list, columns=name_list).to_csv('radiomics_outputs_analysis/rv2_standardized.csv')

